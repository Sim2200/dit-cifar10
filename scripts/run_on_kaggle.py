"""Run an experiment on Kaggle's free 2x T4 and pull the results back.

1. Packages src/ and kaggle/ into a private Kaggle dataset <user>/dit-cifar10-src
   (kernels can only attach datasets, not git repos).
2. Pushes kaggle/<script> as a private GPU script kernel with internet on (CIFAR-10 download),
   polls until it finishes, then downloads /kaggle/working/results/* into results/.

    python scripts/run_on_kaggle.py --user simrankharbanda22 --experiment pilot
    python scripts/run_on_kaggle.py --user simrankharbanda22 --experiment pilot --pull-only
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

EXPERIMENTS = {"pilot": "run_pilot.py"}


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


def push_kernel(user: str, src_slug: str, experiment: str) -> str:
    script = EXPERIMENTS[experiment]
    kid = f"{user}/{PROJECT}-{experiment}"
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        shutil.copy(ROOT / "kaggle" / script, d / script)
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": kid, "title": kid.split("/")[1], "code_file": script, "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [src_slug], "competition_sources": [], "kernel_sources": []}))
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
    ap.add_argument("--pull-only", action="store_true")
    a = ap.parse_args()
    kid = f"{a.user}/{PROJECT}-{a.experiment}"
    if not a.pull_only:
        slug = push_source(a.user)
        kid = push_kernel(a.user, slug, a.experiment)
        print("final:", wait(kid))
    pull(kid)


if __name__ == "__main__":
    main()
