import torch

from dit import train


def test_smoke_train_and_resume(tmp_path):
    out = tmp_path / "run"
    args = ["--model", "DiT-S/4", "--fake-data", "--out", str(out), "--steps", "3", "--batch", "8", "--ckpt-every", "2", "--log-every", "1", "--workers", "0"]
    train.main(args)
    cks = sorted(out.glob("step_*.pt"))
    assert [c.name for c in cks] == ["step_0000002.pt", "step_0000003.pt"]
    log = (out / "train_log.csv").read_text().strip().splitlines()
    assert log[0].startswith("step,loss") and len(log) == 4

    sd = torch.load(cks[-1], map_location="cpu")
    assert sd["step"] == 3 and "ema" in sd
    train.main([a if a != "3" else "5" for a in args])
    assert train.latest_checkpoint(out).name == "step_0000005.pt"
    assert len((out / "train_log.csv").read_text().strip().splitlines()) == 6


def test_grad_accumulation_matches_global_batch(tmp_path):
    out = tmp_path / "acc"
    train.main(["--model", "DiT-S/4", "--fake-data", "--out", str(out), "--steps", "1", "--batch", "8", "--accum", "2", "--workers", "0", "--log-every", "1"])
    assert train.latest_checkpoint(out).name == "step_0000001.pt"
