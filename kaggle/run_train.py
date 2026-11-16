"""One training run on Kaggle, resumable across sessions. Pushed by
scripts/run_on_kaggle.py --experiment train --model DiT-S/4 ..., which fills CONFIG.

If a previous version of this kernel is attached as a kernel source, its latest checkpoint for
this run is copied into /kaggle/working first and training resumes from it. Stops cleanly at
MAX_HOURS so the session never hits Kaggle's 12 h kill; the runner re-pushes until done.

Writes /kaggle/working/checkpoints/<run>/ (last checkpoints + EMA, train_log.csv, config.json)
and /kaggle/working/results/train_<run>.json.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CONFIG = {"run": "s4", "model": "DiT-S/4", "gpus": 1, "steps": 100000, "batch": 128, "precision": "fp16", "max_hours": 10.5, "ckpt_every": 5000}

WORK = Path("/kaggle/working")
RES = WORK / "results"
CK = WORK / "checkpoints" / CONFIG["run"]
RES.mkdir(parents=True, exist_ok=True)
CK.mkdir(parents=True, exist_ok=True)
SRC = Path(glob.glob("/kaggle/input/**/src/dit/model.py", recursive=True)[0]).parents[2]
os.environ["PYTHONPATH"] = str(SRC / "src")
PY = sys.executable
DATA = WORK / "data"


def restore_previous():
    prev = sorted(glob.glob(f"/kaggle/input/**/checkpoints/{CONFIG['run']}/step_*.pt", recursive=True))
    if not prev:
        print("no previous checkpoint; starting fresh", flush=True)
        return 0
    latest = max(prev, key=lambda p: int(Path(p).stem.split("_")[1]))
    shutil.copy(latest, CK / Path(latest).name)
    log = Path(latest).parent / "train_log.csv"
    if log.exists():
        shutil.copy(log, CK / "train_log.csv")
    step = int(Path(latest).stem.split("_")[1])
    print(f"restored {latest} (step {step})", flush=True)
    return step


def main():
    t0 = time.time()
    start_step = restore_previous()
    subprocess.run([PY, "-c", f"from torchvision import datasets; datasets.CIFAR10('{DATA}', train=True, download=True)"], check=True)
    base = ["-m", "dit.train", "--model", CONFIG["model"], "--data", str(DATA), "--out", str(CK), "--steps", str(CONFIG["steps"]), "--batch", str(CONFIG["batch"]), "--precision", CONFIG["precision"], "--log-every", "100", "--ckpt-every", str(CONFIG["ckpt_every"]), "--keep", "2", "--workers", "4", "--max-hours", str(CONFIG["max_hours"])]
    cmd = [PY, "-m", "torch.distributed.run", "--nproc_per_node", str(CONFIG["gpus"]), *base] if CONFIG["gpus"] > 1 else [PY, *base]
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    cks = sorted(CK.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    end_step = int(cks[-1].stem.split("_")[1]) if cks else start_step
    rec = dict(config=CONFIG, start_step=start_step, end_step=end_step, finished=end_step >= CONFIG["steps"], exit_code=r.returncode, session_hours=round((time.time() - t0) / 3600, 2))
    if r.returncode == 0:
        rec["checkpoint"] = cks[-1].name if cks else None
    print(json.dumps(rec, indent=1), flush=True)
    (RES / f"train_{CONFIG['run']}.json").write_text(json.dumps(rec, indent=2))
    if r.returncode != 0:
        raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
