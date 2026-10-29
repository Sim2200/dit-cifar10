import pytest
import torch

from dit.model import DiT, DIT_CONFIGS, build_dit, count_params, forward_gflops

torch.manual_seed(0)


def small():
    return DiT(input_size=32, patch_size=4, hidden_size=64, depth=2, num_heads=4)


def test_output_shape_learn_sigma():
    m = small().eval()
    x = torch.randn(3, 3, 32, 32)
    out = m(x, torch.randint(0, 1000, (3,)), torch.randint(0, 10, (3,)))
    assert out.shape == (3, 6, 32, 32)


def test_output_shape_fixed_sigma():
    m = DiT(patch_size=4, hidden_size=64, depth=2, num_heads=4, learn_sigma=False).eval()
    x = torch.randn(2, 3, 32, 32)
    assert m(x, torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long)).shape == (2, 3, 32, 32)


def test_block_is_identity_at_init():
    m = small().eval()
    x = torch.randn(2, m.num_patches, 64)
    c = torch.randn(2, 64)
    for block in m.blocks:
        assert torch.allclose(block(x, c), x)


def test_final_output_is_zero_at_init():
    m = small().eval()
    out = m(torch.randn(2, 3, 32, 32), torch.randint(0, 1000, (2,)), torch.randint(0, 10, (2,)))
    assert torch.all(out == 0)


def test_adaln_produces_six_chunks():
    m = small()
    c = torch.randn(2, 64)
    mod = m.blocks[0].adaLN_modulation(c)
    assert mod.shape == (2, 6 * 64)
    assert len(mod.chunk(6, dim=1)) == 6


def test_unpatchify_inverts_patch_layout():
    m = DiT(patch_size=4, hidden_size=64, depth=1, num_heads=4, learn_sigma=False)
    img = torch.randn(2, 3, 32, 32)
    p = m.patch_size
    tokens = img.reshape(2, 3, 8, p, 8, p).permute(0, 2, 4, 3, 5, 1).reshape(2, 64, p * p * 3)
    assert torch.equal(m.unpatchify(tokens), img)


def test_pos_embed_is_fixed_and_nonzero():
    m = small()
    assert not m.pos_embed.requires_grad
    assert m.pos_embed.abs().sum() > 0


def test_label_dropout_uses_null_class():
    m = small().train()
    m.y_embedder.dropout_prob = 1.0
    y = torch.arange(10)
    dropped = m.y_embedder.token_drop(y)
    assert torch.all(dropped == 10)


def test_cfg_output_shape_and_variance_passthrough():
    m = small().eval()
    for p in m.final_layer.linear.parameters():
        torch.nn.init.normal_(p, std=0.02)
    x = torch.randn(4, 3, 32, 32)
    y = torch.cat([torch.zeros(2, dtype=torch.long), torch.full((2,), 10, dtype=torch.long)])
    out = m.forward_with_cfg(x, torch.full((4,), 500, dtype=torch.long), y, cfg_scale=2.0)
    assert out.shape == (4, 6, 32, 32)
    assert torch.allclose(out[:2, :3], out[2:, :3])


@pytest.mark.parametrize(
    "name,params_m,gflops",
    [("DiT-S/4", 33, 1.4), ("DiT-S/2", 33, 6.1), ("DiT-B/4", 130, 5.6), ("DiT-B/2", 130, 23.0)],
)
def test_param_counts_and_gflops_match_paper_table(name, params_m, gflops):
    m = build_dit(name)
    assert abs(count_params(m) / 1e6 - params_m) / params_m < 0.03
    assert abs(forward_gflops(m) - gflops) / gflops < 0.10


def test_one_step_reduces_loss_on_fixed_batch():
    from dit.diffusion import GaussianDiffusion

    torch.manual_seed(0)
    m = small().train()
    diff = GaussianDiffusion()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0)
    x = torch.rand(8, 3, 32, 32) * 2 - 1
    y = torch.randint(0, 10, (8,))
    t = torch.randint(0, 1000, (8,))
    torch.manual_seed(1)
    before = diff.training_losses(m, x, t, dict(y=y))["loss"].mean()
    before.backward()
    opt.step()
    opt.zero_grad()
    for _ in range(4):
        torch.manual_seed(1)
        diff.training_losses(m, x, t, dict(y=y))["loss"].mean().backward()
        opt.step()
        opt.zero_grad()
    torch.manual_seed(1)
    after = diff.training_losses(m, x, t, dict(y=y))["loss"].mean()
    assert after < before
