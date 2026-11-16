"""One interface over the samplers and the two denoisers.

Denoisers: our DiT (epsilon + learned variance, class-conditional with CFG) and the pretrained
DDPM U-Net google/ddpm-cifar10-32 (epsilon, fixed variance, unconditional) through diffusers.
Samplers: DDPM ancestral (ours), DDIM eta 0 (ours), DPM-Solver++ 2M (diffusers scheduler
wrapped around either denoiser's epsilon prediction).
"""

import time

import torch

from .diffusion import GaussianDiffusion


class DiTDenoiser:
    def __init__(self, model, cfg_scale=1.0, num_classes=10):
        self.model = model.eval()
        self.cfg_scale = cfg_scale
        self.num_classes = num_classes
        self.learn_sigma = model.learn_sigma
        self.in_channels = model.in_channels

    def __call__(self, x, t, y):
        if self.cfg_scale == 1.0:
            return self.model(x, t, y)
        null = torch.full_like(y, self.num_classes)
        out = self.model.forward_with_cfg(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, null]), self.cfg_scale)
        return out[: x.shape[0]]

    def eps(self, x, t, y):
        return self(x, t, y)[:, : self.in_channels]


class UNetDenoiser:
    """google/ddpm-cifar10-32: unconditional, predicts epsilon only. Labels are ignored."""

    def __init__(self, unet):
        self.model = unet.eval()
        self.learn_sigma = False
        self.in_channels = 3

    def __call__(self, x, t, y=None):
        return self.model(x, t).sample

    def eps(self, x, t, y=None):
        return self(x, t, y)

    @classmethod
    def pretrained(cls, device, repo="google/ddpm-cifar10-32"):
        from diffusers import UNet2DModel

        return cls(UNet2DModel.from_pretrained(repo).to(device))


def dpm_solver_pp(denoiser, shape, y, device, num_steps, generator=None, order=2):
    from diffusers import DPMSolverMultistepScheduler

    sched = DPMSolverMultistepScheduler(
        num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02, beta_schedule="linear",
        solver_order=order, prediction_type="epsilon", algorithm_type="dpmsolver++", thresholding=False,
    )
    sched.set_timesteps(num_steps, device=device)
    x = torch.randn(shape, device=device, generator=generator)
    for t in sched.timesteps:
        tt = torch.full((shape[0],), int(t), device=device, dtype=torch.long)
        eps = denoiser.eps(x, tt, y)
        x = sched.step(eps, t, x).prev_sample
    return x


@torch.no_grad()
def sample(denoiser, sampler, num_steps, y, device, generator=None, diffusion=None, amp=True):
    """Returns images in [-1, 1] and wall seconds. sampler in {ddpm, ddim, dpmpp}."""
    diffusion = diffusion or GaussianDiffusion(learn_sigma=denoiser.learn_sigma)
    shape = (y.shape[0], 3, 32, 32)
    kw = dict(y=y)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
        if sampler == "ddpm":
            assert num_steps == diffusion.timesteps
            x = diffusion.p_sample_loop(denoiser, shape, kw, device, generator=generator)
        elif sampler == "ddim":
            x = diffusion.ddim_sample_loop(denoiser, shape, kw, device, num_steps=num_steps, generator=generator)
        elif sampler == "dpmpp":
            x = dpm_solver_pp(denoiser, shape, y, device, num_steps, generator=generator)
        else:
            raise ValueError(sampler)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return x.float().clamp(-1, 1), time.time() - t0


def to_uint8(x):
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
