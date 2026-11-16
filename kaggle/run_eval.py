"""Sampler sweep and metrics for one trained run (or the U-Net baseline) on a Kaggle T4.
Pushed by scripts/run_on_kaggle.py --experiment eval --run s4, which fills CONFIG and attaches
the training kernel's output so the EMA checkpoint is under /kaggle/input.

For each (sampler, steps) setting: generate N class-balanced images, record seconds per image,
compute FID / IS / precision / recall against CIFAR-10 train statistics, class accuracy under a
pretrained ResNet-56, and the memorisation check. Writes /kaggle/working/results/eval_<run>.json
plus a 10x10 sample grid per setting. Sample sets (about 29 MB each) stay in the output so a
re-run after a failure skips the sampling.
"""

import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CONFIG = {"run": "s4", "unet": False, "n": 10000, "cfg": 1.0, "settings": [["ddpm", 1000], ["ddim", 250], ["ddim", 50], ["ddim", 10], ["ddim", 4], ["dpmpp", 50], ["dpmpp", 10], ["dpmpp", 4]], "cfg_sweep": [1.0, 1.5, 2.0, 4.0], "cfg_sweep_setting": ["ddim", 50]}

WORK = Path("/kaggle/working")
RES = WORK / "results"
FIG = RES / "figures"
SAMPLES = WORK / "samples"
for d in (RES, FIG, SAMPLES):
    d.mkdir(parents=True, exist_ok=True)
SRC = Path(glob.glob("/kaggle/input/**/src/dit/model.py", recursive=True)[0]).parents[2]
os.environ["PYTHONPATH"] = str(SRC / "src")
PY = sys.executable
DATA, CACHE = WORK / "data", WORK / "cache"


def sh(cmd):
    print("$", " ".join(str(c) for c in cmd)[:300], flush=True)
    t0 = time.time()
    r = subprocess.run([str(c) for c in cmd], text=True, capture_output=True)
    print(f"  -> exit {r.returncode} in {time.time() - t0:.0f} s", flush=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
        raise SystemExit(f"failed: {cmd[:4]}")
    return r.stdout


def grid(npz, png):
    import numpy as np
    from PIL import Image

    z = np.load(npz)
    imgs, labels = z["images"], z["labels"]
    rows = [np.concatenate([imgs[np.where(labels == c)[0][i]] for i in range(10)], axis=1) for c in range(10)]
    Image.fromarray(np.concatenate(rows, axis=0)).save(png)


def restore_previous():
    """Reuse a previous version's output: CIFAR-10 archive, feature cache, finished sample sets."""
    import shutil

    roots = {Path(p).parents[1] for p in glob.glob("/kaggle/input/**/samples/*.npz", recursive=True)} | {Path(p).parents[1] for p in glob.glob("/kaggle/input/**/cache/*.pt", recursive=True)}
    for root in roots:
        for sub in ("data", "cache", "samples"):
            src = root / sub
            if src.is_dir():
                shutil.copytree(src, WORK / sub, dirs_exist_ok=True)
                print(f"restored {sub}/ from {root}", flush=True)


def main():
    t_all = time.time()
    restore_previous()
    sh([PY, "-m", "pip", "install", "-q", "torch-fidelity", "diffusers"])
    ckpt = None
    if not CONFIG["unet"]:
        cks = glob.glob(f"/kaggle/input/**/checkpoints/{CONFIG['run']}/step_*.pt", recursive=True)
        ckpt = max(cks, key=lambda p: int(Path(p).stem.split("_")[1]))
        print("checkpoint:", ckpt, flush=True)
    tag = "unet" if CONFIG["unet"] else CONFIG["run"]
    source = ["--unet"] if CONFIG["unet"] else ["--ckpt", ckpt]

    jobs = [(s, k, CONFIG["cfg"]) for s, k in CONFIG["settings"]]
    if not CONFIG["unet"]:
        s0, k0 = CONFIG["cfg_sweep_setting"]
        jobs += [(s0, k0, c) for c in CONFIG["cfg_sweep"] if c != CONFIG["cfg"]]

    results = []
    for sampler, steps, cfg in jobs:
        name = f"{tag}_{sampler}{steps}_cfg{cfg}"
        npz = SAMPLES / f"{name}.npz"
        if not npz.exists():
            sh([PY, "-m", "dit.sample", *source, "--sampler", sampler, "--steps", steps, "--cfg", cfg, "--n", CONFIG["n"], "--batch", 500, "--out", npz])
        out = RES / f"eval_{name}.json"
        sh([PY, "-m", "dit.evaluate", npz, "--data", DATA, "--cache", CACHE, "--out", out])
        rec = json.loads(out.read_text())
        grid(npz, FIG / f"{name}.png")
        results.append(rec)
        print(json.dumps({k: rec[k] for k in ("fid", "inception_score", "precision", "recall")} | {"s_per_img": rec["meta"]["seconds_per_image"], "setting": name}), flush=True)

    summary = dict(config=CONFIG, checkpoint=ckpt, results=results, total_minutes=round((time.time() - t_all) / 60, 1))
    (RES / f"eval_{tag}.json").write_text(json.dumps(summary, indent=2))
    print("done in", summary["total_minutes"], "min", flush=True)


if __name__ == "__main__":
    main()
