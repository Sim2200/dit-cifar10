"""Gaussian diffusion with a linear beta schedule, epsilon prediction and the
learned-variance (learn-sigma) objective of Improved DDPM (Nichol and Dhariwal, 2021).

Variance is parameterised as an interpolation v in [0, 1] between
log(beta_t) and log(posterior_variance_t), as in the DiT reference.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)


def _extract(arr, t, broadcast_shape):
    out = arr.to(t.device)[t].float()
    while out.dim() < len(broadcast_shape):
        out = out[..., None]
    return out.expand(broadcast_shape)


def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))


def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def discretized_gaussian_log_likelihood(x, means, log_scales):
    """Log-likelihood of images in [-1, 1] quantised to 256 bins."""
    centered = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    return torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )


class GaussianDiffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, learn_sigma=True):
        self.timesteps = timesteps
        self.learn_sigma = learn_sigma
        betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        f = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float64))
        self.betas = f(betas)
        self.alphas_cumprod = f(alphas_cumprod)
        self.alphas_cumprod_prev = f(alphas_cumprod_prev)
        self.sqrt_alphas_cumprod = f(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = f(np.sqrt(1.0 - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = f(np.sqrt(1.0 / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = f(np.sqrt(1.0 / alphas_cumprod - 1))
        self.posterior_variance = f(posterior_variance)
        self.posterior_log_variance_clipped = f(np.log(np.append(posterior_variance[1], posterior_variance[1:])))
        self.posterior_mean_coef1 = f(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.posterior_mean_coef2 = f((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod))

    def q_sample(self, x_start, t, noise):
        return (
            _extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        mean = (
            _extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        return mean, _extract(self.posterior_variance, t, x_t.shape), _extract(self.posterior_log_variance_clipped, t, x_t.shape)

    def predict_xstart_from_eps(self, x_t, t, eps):
        return _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps

    def p_mean_variance(self, model_output, x_t, t, clip_denoised=True):
        C = x_t.shape[1]
        if self.learn_sigma:
            eps, v = torch.split(model_output, C, dim=1)
            min_log = _extract(self.posterior_log_variance_clipped, t, x_t.shape)
            max_log = _extract(torch.log(self.betas), t, x_t.shape)
            frac = (v + 1) / 2
            log_variance = frac * max_log + (1 - frac) * min_log
            variance = torch.exp(log_variance)
        else:
            eps = model_output
            variance = _extract(self.posterior_variance, t, x_t.shape)
            log_variance = _extract(self.posterior_log_variance_clipped, t, x_t.shape)
        x_start = self.predict_xstart_from_eps(x_t, t, eps)
        if clip_denoised:
            x_start = x_start.clamp(-1, 1)
        mean, _, _ = self.q_posterior_mean_variance(x_start, x_t, t)
        return dict(mean=mean, variance=variance, log_variance=log_variance, pred_xstart=x_start, eps=eps)

    def vb_terms_bits(self, model_output, x_start, x_t, t):
        """Variational-bound term at step t in bits per dim, with the mean frozen
        (gradient flows only through the predicted variance), as in Improved DDPM."""
        true_mean, _, true_log_var = self.q_posterior_mean_variance(x_start, x_t, t)
        out = self.p_mean_variance(model_output, x_t, t, clip_denoised=False)
        kl = normal_kl(true_mean, true_log_var, out["mean"], out["log_variance"]).mean(dim=(1, 2, 3)) / math.log(2.0)
        nll = -discretized_gaussian_log_likelihood(x_start, out["mean"], 0.5 * out["log_variance"]).mean(dim=(1, 2, 3)) / math.log(2.0)
        return torch.where(t == 0, nll, kl)

    def training_losses(self, model, x_start, t, model_kwargs):
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        model_output = model(x_t, t, **model_kwargs)
        C = x_start.shape[1]
        terms = {}
        if self.learn_sigma:
            eps, v = torch.split(model_output, C, dim=1)
            frozen = torch.cat([eps.detach(), v], dim=1)
            terms["vb"] = self.vb_terms_bits(frozen, x_start, x_t, t) * (self.timesteps / 1000.0)
        else:
            eps = model_output
        terms["mse"] = F.mse_loss(eps, noise, reduction="none").mean(dim=(1, 2, 3))
        terms["loss"] = terms["mse"] + terms["vb"] if self.learn_sigma else terms["mse"]
        return terms

    @torch.no_grad()
    def p_sample_loop(self, model_fn, shape, model_kwargs, device, progress=False, generator=None):
        """Ancestral DDPM sampling over all T steps. model_fn(x, t, **kwargs) -> model output."""
        x = torch.randn(shape, device=device, generator=generator)
        steps = range(self.timesteps - 1, -1, -1)
        if progress:
            from tqdm import tqdm
            steps = tqdm(steps)
        for i in steps:
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            out = self.p_mean_variance(model_fn(x, t, **model_kwargs), x, t)
            noise = torch.randn(shape, device=device, generator=generator)
            nonzero = (t != 0).float().view(-1, *([1] * (len(shape) - 1)))
            x = out["mean"] + nonzero * torch.exp(0.5 * out["log_variance"]) * noise
        return x

    @torch.no_grad()
    def ddim_sample_loop(self, model_fn, shape, model_kwargs, device, num_steps=50, eta=0.0, generator=None):
        """DDIM (Song et al. 2021) on an evenly spaced subsequence of timesteps."""
        x = torch.randn(shape, device=device, generator=generator)
        seq = np.linspace(0, self.timesteps - 1, num_steps).round().astype(int)[::-1]
        for i, step in enumerate(seq):
            t = torch.full((shape[0],), int(step), device=device, dtype=torch.long)
            out = self.p_mean_variance(model_fn(x, t, **model_kwargs), x, t)
            eps = out["eps"]
            x0 = out["pred_xstart"]
            alpha_bar = _extract(self.alphas_cumprod, t, x.shape)
            if i + 1 < len(seq):
                t_prev = torch.full((shape[0],), int(seq[i + 1]), device=device, dtype=torch.long)
                alpha_bar_prev = _extract(self.alphas_cumprod, t_prev, x.shape)
            else:
                alpha_bar_prev = torch.ones_like(alpha_bar)
            sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
            eps = (x - torch.sqrt(alpha_bar) * x0) / torch.sqrt(1 - alpha_bar)
            mean = torch.sqrt(alpha_bar_prev) * x0 + torch.sqrt((1 - alpha_bar_prev - sigma**2).clamp(min=0)) * eps
            noise = torch.randn(shape, device=device, generator=generator) if i + 1 < len(seq) else 0.0
            x = mean + sigma * noise
        return x
