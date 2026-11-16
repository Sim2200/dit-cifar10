"""Run an experiment on Kaggle's free 2x T4 and pull the results back.

1. Packages src/ and kaggle/ into a private Kaggle dataset <user>/dit-cifar10-src
   (kernels can only attach datasets, not git repos).
2. Pushes kaggle/<script> as a private GPU script kernel with internet on (CIFAR-10 download),
   polls until it finishes, then downloads /kaggle/working/results/* into results/.

    python scripts/run_on_kaggle.py --user simrankharbanda22 --experiment pilot
    python scripts/run_on_kaggle.py --user simrankharbanda22 --experiment pilot --pull-only
    python scripts/run_on_kaggle.py --user simran2200 --experiment train --run s4   # re-run to resume
Account selection: KAGGLE_API_TOKEN=$(cat ~/.kaggle-2/access_token) for the second account.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = [sys.executable, "-m", "kaggle"]
PROJECT = "dit-cifar10"

EXPERIMENTS = {"pilot": "run_pilot.py", "train": "run_train.py"}

TRAIN_RUNS = {  # run -> settings; gpus and hours from results/pilot.json
    "s4": dict(model="DiT-S/4", gpus=1, steps=100_000, batch=128, precision="fp16", max_hours=10.5, ckpt_every=5000),
    "s2": dict(model="DiT-S/2", gpus=2, steps=100_000, batch=128, precision="fp16", max_hours=10.5, ckpt_every=5000),
    "b4": dict(model="DiT-B/4", gpus=2, steps=100_000, batch=128, precision="fp16", max_hours=10.5, ckpt_every=5000),
}


def kaggle(*args: str, check: bool = True) -> str:
    r = subprocess.run([*KAGGLE, *args], text=True, capture_output=True)
    if check and r.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout + r.stderr


def push_source(user: str) -> str:
    slug = f"{user}/{PROJECT}-src"
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for sub in ("src", "kaggle", "tests"):
            shutil.copytree(ROOT / sub, d / sub, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "out"))
        (d / "dataset-metadata.json").write_text(json.dumps({"title": f"{PROJECT}-src", "id": slug, "licenses": [{"name": "CC0-1.0"}]}))
        exists = f"{PROJECT}-src" in kaggle("datasets", "list", "--mine", check=False)
        out = kaggle("datasets", "version" if exists else "create", "-p", str(d), *(["-m", "update"] if exists else []), "--dir-mode", "zip", check=False)
        print(out.strip().splitlines()[-1])
    time.sleep(20)
    return slug


def kernel_exists(kid: str) -> bool:
    return kid.split("/")[1] in kaggle("kernels", "list", "--mine", "--page-size", "200", check=False)


def push_kernel(user: str, src_slug: str, experiment: str, run: str = "") -> str:
    script = EXPERIMENTS[experiment]
    kid = f"{user}/{PROJECT}-{experiment}" + (f"-{run}" if run else "")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        text = (ROOT / "kaggle" / script).read_text()
        if run:
            cfg = dict(run=run, **TRAIN_RUNS[run])
            head, _, rest = text.partition("\nCONFIG = ")
            text = head + "\nCONFIG = " + json.dumps(cfg) + rest[rest.index("\n"):]
        (d / script).write_text(text)
        # a previous version's output holds the checkpoint to resume from
        sources = [kid] if run and kernel_exists(kid) else []
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": kid, "title": kid.split("/")[1], "code_file": script, "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [src_slug], "competition_sources": [], "kernel_sources": sources}))
        print(kaggle("kernels", "push", "-p", str(d)).strip().splitlines()[-1])
    return kid


def wait(kid: str, poll: int = 60) -> str:
    t0 = time.time()
    while time.time() - t0 < 300:
        status = kaggle("kernels", "status", kid, check=False).strip().splitlines()[-1].lower()
        if "queued" in status or "running" in status:
            break
        time.sleep(10)
    while True:
        status = kaggle("kernels", "status", kid, check=False).strip().splitlines()[-1]
        print(f"  {time.time() - t0:6.0f} s  {status}", flush=True)
        if any(k in status.lower() for k in ("complete", "error", "cancel")):
            return status
        time.sleep(poll)


def pull(kid: str) -> None:
    out = ROOT / "kaggle" / "out"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    print(kaggle("kernels", "output", kid, "-p", str(out)).strip().splitlines()[-1])
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    for f in out.rglob("*"):
        if f.suffix in (".json", ".png", ".csv", ".md") and f.is_file():
            shutil.copy(f, results / f.name)
            print("  pulled", f.name)
    log = next(out.glob("*.log"), None)
    if log:
        lines = []
        for line in log.read_text().splitlines():
            line = line.strip().lstrip(",")
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("stream_name") == "stdout" or "rror" in o.get("data", ""):
                lines.append(o["data"].rstrip())
        (results / f"kaggle_{kid.split('-')[-1]}.log").write_text("\n".join(lines))
        print("\n".join(lines[-30:]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--experiment", default="pilot", choices=sorted(EXPERIMENTS))
    ap.add_argument("--run", default="", choices=["", *TRAIN_RUNS], help="for --experiment train")
    ap.add_argument("--pull-only", action="store_true")
    ap.add_argument("--no-wait", action="store_true", help="push and return without polling")
    a = ap.parse_args()
    if a.experiment == "train" and not a.run:
        raise SystemExit("--experiment train needs --run")
    kid = f"{a.user}/{PROJECT}-{a.experiment}" + (f"-{a.run}" if a.run else "")
    if not a.pull_only:
        slug = push_source(a.user)
        kid = push_kernel(a.user, slug, a.experiment, a.run)
        if a.no_wait:
            print("pushed", kid)
            return
        print("final:", wait(kid))
    pull(kid)


if __name__ == "__main__":
    main()
