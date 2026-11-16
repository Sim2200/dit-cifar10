import json

import numpy as np
import torch

from dit import sample as sample_cli
from dit.diffusion import GaussianDiffusion
from dit.model import DiT
from dit.samplers import DiTDenoiser, dpm_solver_pp, sample, to_uint8


def tiny():
    m = DiT(patch_size=4, hidden_size=32, depth=1, num_heads=4).eval()
    for p in m.final_layer.linear.parameters():
        torch.nn.init.normal_(p, std=0.02)
    for b in m.blocks:
        torch.nn.init.normal_(b.adaLN_modulation[-1].weight, std=0.02)
    return m


def test_dit_denoiser_eps_channels():
    den = DiTDenoiser(tiny())
    x = torch.randn(2, 3, 32, 32)
    assert den.eps(x, torch.tensor([5, 5]), torch.tensor([0, 1])).shape == (2, 3, 32, 32)


def test_cfg_denoiser_changes_output():
    m = tiny()
    x = torch.randn(2, 3, 32, 32)
    t, y = torch.tensor([500, 500]), torch.tensor([0, 1])
    a = DiTDenoiser(m, cfg_scale=1.0).eps(x, t, y)
    b = DiTDenoiser(m, cfg_scale=4.0).eps(x, t, y)
    assert a.shape == b.shape and not torch.allclose(a, b)


def test_dpmpp_runs_on_dit():
    den = DiTDenoiser(tiny())
    x = dpm_solver_pp(den, (2, 3, 32, 32), torch.tensor([0, 1]), torch.device("cpu"), num_steps=4, generator=torch.Generator().manual_seed(0))
    assert x.shape == (2, 3, 32, 32) and torch.isfinite(x).all()


def test_sample_dispatch_all_samplers():
    den = DiTDenoiser(tiny())
    d = GaussianDiffusion(timesteps=10)
    dev = torch.device("cpu")
    y = torch.tensor([3, 7])
    for s, n in (("ddpm", 10), ("ddim", 4), ("dpmpp", 4)):
        x, dt = sample(den, s, n, y, dev, generator=torch.Generator().manual_seed(0), diffusion=d if s != "dpmpp" else None)
        assert x.shape == (2, 3, 32, 32) and x.min() >= -1 and x.max() <= 1 and dt >= 0


def test_to_uint8_range():
    u = to_uint8(torch.tensor([-1.0, 0.0, 1.0, 2.0]))
    assert u.tolist() == [0, 128, 255, 255]


def test_sample_cli_writes_npz(tmp_path):
    from dit.ema import EMA

    m = tiny()
    ck = tmp_path / "ck.pt"
    torch.save(dict(step=7, model=m.state_dict(), ema=EMA(m).state_dict(), args=dict(model="DiT-S/4")), ck)
    # the checkpoint names DiT-S/4 but holds tiny weights, so load through a patched builder
    orig = sample_cli.build_dit
    sample_cli.build_dit = lambda name, **kw: DiT(patch_size=4, hidden_size=32, depth=1, num_heads=4)
    try:
        out = tmp_path / "s.npz"
        meta = sample_cli.main(["--ckpt", str(ck), "--sampler", "ddim", "--steps", "2", "--n", "6", "--batch", "3", "--out", str(out)])
    finally:
        sample_cli.build_dit = orig
    z = np.load(out)
    assert z["images"].shape == (6, 32, 32, 3) and z["images"].dtype == np.uint8
    assert z["labels"].tolist() == [0, 1, 2, 3, 4, 5]
    assert json.loads(str(z["meta"]))["step"] == 7 and meta["seconds_per_image"] > 0
