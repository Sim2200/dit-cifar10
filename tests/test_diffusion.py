import math

import numpy as np
import torch

from dit.diffusion import GaussianDiffusion, normal_kl
from dit.ema import EMA
from dit.model import DiT


def test_schedule_matches_ddpm_constants():
    d = GaussianDiffusion()
    assert d.betas[0].item() == 1e-4 and abs(d.betas[-1].item() - 0.02) < 1e-12
    assert abs(d.alphas_cumprod[-1].item() - 4.035e-5) < 1e-6
    assert torch.all(d.posterior_variance >= 0)


def test_q_sample_marginal_variance():
    d = GaussianDiffusion()
    x0 = torch.zeros(1, 3, 32, 32)
    t = torch.tensor([999])
    xt = d.q_sample(x0, t, torch.ones_like(x0))
    assert torch.allclose(xt, torch.full_like(x0, d.sqrt_one_minus_alphas_cumprod[999].item()), atol=1e-6)


def test_predict_xstart_inverts_q_sample():
    d = GaussianDiffusion()
    x0 = torch.rand(2, 3, 32, 32) * 2 - 1
    t = torch.tensor([10, 900])
    eps = torch.randn_like(x0)
    assert torch.allclose(d.predict_xstart_from_eps(d.q_sample(x0, t, eps), t, eps), x0, atol=1e-4)


def test_normal_kl_zero_for_identical():
    m = torch.randn(4)
    lv = torch.randn(4)
    assert torch.allclose(normal_kl(m, lv, m, lv), torch.zeros(4), atol=1e-6)


def test_vb_term_matches_closed_form_at_true_posterior():
    d = GaussianDiffusion()
    x0 = torch.rand(2, 3, 32, 32) * 2 - 1
    t = torch.tensor([500, 500])
    eps = torch.randn_like(x0)
    xt = d.q_sample(x0, t, eps)
    v = torch.full_like(x0, -1.0)
    vb = d.vb_terms_bits(torch.cat([eps, v], dim=1), x0, xt, t)
    assert torch.allclose(vb, torch.zeros(2), atol=1e-4)


def test_learned_variance_interpolates_between_bounds():
    d = GaussianDiffusion()
    xt = torch.randn(1, 3, 32, 32)
    t = torch.tensor([300])
    eps = torch.zeros_like(xt)
    lo = d.p_mean_variance(torch.cat([eps, torch.full_like(xt, -1.0)], 1), xt, t)["log_variance"]
    hi = d.p_mean_variance(torch.cat([eps, torch.full_like(xt, 1.0)], 1), xt, t)["log_variance"]
    assert torch.allclose(lo, torch.full_like(lo, d.posterior_log_variance_clipped[300].item()), atol=1e-5)
    assert torch.allclose(hi, torch.full_like(hi, math.log(d.betas[300].item())), atol=1e-5)


def test_training_losses_keys_and_shapes():
    m = DiT(patch_size=4, hidden_size=32, depth=1, num_heads=4).train()
    d = GaussianDiffusion()
    terms = d.training_losses(m, torch.randn(4, 3, 32, 32), torch.randint(0, 1000, (4,)), dict(y=torch.zeros(4, dtype=torch.long)))
    assert set(terms) == {"mse", "vb", "loss"}
    assert terms["loss"].shape == (4,)
    assert torch.isfinite(terms["loss"]).all()


def test_fixed_sigma_loss_has_no_vb():
    m = DiT(patch_size=4, hidden_size=32, depth=1, num_heads=4, learn_sigma=False).train()
    d = GaussianDiffusion(learn_sigma=False)
    terms = d.training_losses(m, torch.randn(2, 3, 32, 32), torch.tensor([0, 999]), dict(y=torch.zeros(2, dtype=torch.long)))
    assert "vb" not in terms and torch.equal(terms["loss"], terms["mse"])


def test_samplers_run_and_return_finite():
    m = DiT(patch_size=4, hidden_size=32, depth=1, num_heads=4).eval()
    d = GaussianDiffusion(timesteps=20)
    kw = dict(y=torch.zeros(2, dtype=torch.long))
    g = torch.Generator().manual_seed(0)
    x = d.p_sample_loop(m, (2, 3, 32, 32), kw, "cpu", generator=g)
    assert x.shape == (2, 3, 32, 32) and torch.isfinite(x).all()
    x = d.ddim_sample_loop(m, (2, 3, 32, 32), kw, "cpu", num_steps=5, generator=g)
    assert x.shape == (2, 3, 32, 32) and torch.isfinite(x).all()


def test_ddim_is_deterministic_given_noise():
    m = DiT(patch_size=4, hidden_size=32, depth=1, num_heads=4).eval()
    d = GaussianDiffusion(timesteps=20)
    kw = dict(y=torch.zeros(2, dtype=torch.long))
    a = d.ddim_sample_loop(m, (2, 3, 32, 32), kw, "cpu", num_steps=5, generator=torch.Generator().manual_seed(3))
    b = d.ddim_sample_loop(m, (2, 3, 32, 32), kw, "cpu", num_steps=5, generator=torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


def test_ema_moves_toward_model():
    m = DiT(patch_size=4, hidden_size=32, depth=1, num_heads=4)
    ema = EMA(m, decay=0.5)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    ema.update(m)
    ema_params = dict(ema.module.named_parameters())
    for name, pm in m.named_parameters():
        if pm.requires_grad:
            assert torch.allclose(ema_params[name], pm - 0.5)
        else:
            assert torch.allclose(ema_params[name], pm - 1.0, atol=1e-6)
