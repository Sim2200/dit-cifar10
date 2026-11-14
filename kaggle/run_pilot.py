"""Throughput pilot on Kaggle's 2x T4. Pushed by scripts/run_on_kaggle.py --experiment pilot.

For each configuration, trains a short run on real CIFAR-10 through the real trainer
(dataloader, EMA, clipping, logging) and records images per second and peak memory, so the
budget for the 100k-step grid comes from measurement rather than Gflops. Also times the
samplers on one model so the sweep can be budgeted.

Writes /kaggle/working/results/pilot.json and copies each run's train_log.csv.
"""

import csv
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
RES = WORK / "results"
RES.mkdir(parents=True, exist_ok=True)
SRC = Path(glob.glob("/kaggle/input/**/src/dit/model.py", recursive=True)[0]).parents[2]
os.environ["PYTHONPATH"] = str(SRC / "src")
PY = sys.executable
STEPS = int(os.environ.get("PILOT_STEPS", "300"))
LOG_EVERY = 50
DATA = WORK / "data"

RUNS = [
    # name, model, gpus, precision, batch, accum
    ("s4_fp32_1gpu", "DiT-S/4", 1, "fp32", 128, 1),
    ("s4_fp16_1gpu", "DiT-S/4", 1, "fp16", 128, 1),
    ("s2_fp32_1gpu", "DiT-S/2", 1, "fp32", 128, 1),
    ("s2_fp16_1gpu", "DiT-S/2", 1, "fp16", 128, 1),
    ("b4_fp16_1gpu", "DiT-B/4", 1, "fp16", 128, 1),
    ("s2_fp16_2gpu", "DiT-S/2", 2, "fp16", 128, 1),
    ("b4_fp16_2gpu", "DiT-B/4", 2, "fp16", 128, 1),
]


def sh(cmd, env=None):
    print("$", " ".join(cmd)[:300], flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, text=True, env={**os.environ, **(env or {})}, capture_output=True)
    print(f"  -> exit {r.returncode} in {time.time() - t0:.0f} s", flush=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    return r


def steady_state(log_path):
    rows = list(csv.DictReader(open(log_path)))
    if len(rows) < 2:
        return None
    tail = rows[1:]  # first window includes warmup and cudnn autotune
    ips = sum(float(r["img_per_s"]) for r in tail) / len(tail)
    mem = max(float(r["peak_mem_gb"]) for r in rows)
    return dict(img_per_s=round(ips, 1), peak_mem_gb=round(mem, 2), final_loss=round(float(rows[-1]["loss"]), 4), windows=len(tail))


def train_run(name, model, gpus, precision, batch, accum):
    out = WORK / "pilot" / name
    base = ["-m", "dit.train", "--model", model, "--data", str(DATA), "--out", str(out), "--steps", str(STEPS), "--batch", str(batch), "--accum", str(accum), "--precision", precision, "--log-every", str(LOG_EVERY), "--ckpt-every", str(STEPS), "--workers", "4"]
    cmd = [PY, "-m", "torch.distributed.run", "--nproc_per_node", str(gpus), *base] if gpus > 1 else [PY, *base]
    env = {"CUDA_VISIBLE_DEVICES": "0" if gpus == 1 else "0,1"}
    r = sh(cmd, env)
    rec = dict(name=name, model=model, gpus=gpus, precision=precision, batch=batch, accum=accum, steps=STEPS, ok=r.returncode == 0)
    if r.returncode == 0:
        rec.update(steady_state(out / "train_log.csv") or {})
        if "img_per_s" in rec:
            rec["steps_per_s"] = round(rec["img_per_s"] / batch, 3)
            rec["hours_per_100k_steps"] = round(100_000 / rec["steps_per_s"] / 3600, 1)
        (RES / f"train_log_{name}.csv").write_bytes((out / "train_log.csv").read_bytes())
    else:
        rec["error"] = (r.stderr or r.stdout)[-1500:]
        if "out of memory" in rec["error"].lower() and accum == 1:
            print("  OOM, retrying with --accum 2", flush=True)
            return train_run(name + "_accum2", model, gpus, precision, batch, 2)
    print(json.dumps(rec, indent=1), flush=True)
    return rec


def sampler_timing():
    import torch

    sys.path.insert(0, str(SRC / "src"))
    from dit.diffusion import GaussianDiffusion
    from dit.model import build_dit

    dev = torch.device("cuda:0")
    out = []
    for name in ("DiT-S/4", "DiT-S/2", "DiT-B/4"):
        m = build_dit(name).to(dev).eval()
        d = GaussianDiffusion()
        n = 100
        y = torch.randint(0, 10, (n,), device=dev)
        with torch.autocast("cuda", dtype=torch.float16):
            d.ddim_sample_loop(m, (n, 3, 32, 32), dict(y=y), dev, num_steps=2)  # warmup
            torch.cuda.synchronize()
            for sampler, steps in (("ddim", 50), ("ddim", 10), ("ddpm", 1000)):
                t0 = time.time()
                if sampler == "ddpm":
                    d.p_sample_loop(m, (n, 3, 32, 32), dict(y=y), dev)
                else:
                    d.ddim_sample_loop(m, (n, 3, 32, 32), dict(y=y), dev, num_steps=steps)
                torch.cuda.synchronize()
                dt = time.time() - t0
                rec = dict(model=name, sampler=sampler, steps=steps, batch=n, seconds_per_image=round(dt / n, 4), images_per_s=round(n / dt, 1), hours_per_10k=round(10_000 * dt / n / 3600, 2))
                print(json.dumps(rec), flush=True)
                out.append(rec)
        del m
        torch.cuda.empty_cache()
    return out


def main():
    t_all = time.time()
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True, capture_output=True)
    import torch
    env = dict(gpus=r.stdout.strip().splitlines(), torch=torch.__version__, cuda=torch.version.cuda, bf16_supported=torch.cuda.is_bf16_supported(), python=sys.version.split()[0])
    print(json.dumps(env, indent=1), flush=True)

    sh([PY, "-c", f"from torchvision import datasets; datasets.CIFAR10('{DATA}', train=True, download=True)"])

    runs = [train_run(*cfg) for cfg in RUNS]
    samplers = sampler_timing()

    result = dict(env=env, pilot_steps=STEPS, train=runs, samplers=samplers, total_minutes=round((time.time() - t_all) / 60, 1))
    (RES / "pilot.json").write_text(json.dumps(result, indent=2))
    print("done in", result["total_minutes"], "min", flush=True)


if __name__ == "__main__":
    main()
