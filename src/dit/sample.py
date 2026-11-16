"""Generate a class-balanced sample set from an EMA checkpoint or the U-Net baseline.

    python -m dit.sample --ckpt checkpoints/s4/step_0100000.pt --sampler ddim --steps 50 --n 10000 --out samples/s4_ddim50.npz
    python -m dit.sample --unet --sampler ddpm --steps 1000 --n 10000 --out samples/unet_ddpm1000.npz

Saves uint8 NHWC images, labels, and timing. Timing excludes the first (warm-up) batch.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import build_dit
from .samplers import DiTDenoiser, UNetDenoiser, sample, to_uint8


def load_dit(ckpt, device):
    sd = torch.load(ckpt, map_location=device)
    model = build_dit(sd["args"]["model"], learn_sigma=not sd["args"].get("no_learn_sigma", False)).to(device)
    model.load_state_dict(sd["ema"])
    return model, sd["args"]["model"], sd["step"]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt")
    p.add_argument("--unet", action="store_true")
    p.add_argument("--sampler", default="ddim", choices=["ddpm", "ddim", "dpmpp"])
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--batch", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--no-amp", action="store_true")
    a = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.unet:
        den, name, step = UNetDenoiser.pretrained(device), "ddpm-cifar10-32-unet", None
    else:
        model, name, step = load_dit(a.ckpt, device)
        den = DiTDenoiser(model, cfg_scale=a.cfg)

    g = torch.Generator(device=device).manual_seed(a.seed)
    labels = torch.arange(a.n) % 10
    images, times = [], []
    for i in range(0, a.n, a.batch):
        y = labels[i : i + a.batch].to(device)
        x, dt = sample(den, a.sampler, a.steps, y, device, generator=g, amp=not a.no_amp)
        images.append(to_uint8(x).permute(0, 2, 3, 1).cpu().numpy())
        times.append((dt, len(y)))
        print(f"{i + len(y)}/{a.n} {dt / len(y):.4f} s/img", flush=True)
    timed = times[1:] or times
    sec_per_img = sum(t for t, _ in timed) / sum(n for _, n in timed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(model=name, step=step, sampler=a.sampler, steps=a.steps, cfg=a.cfg, n=a.n, batch=a.batch, seed=a.seed, seconds_per_image=sec_per_img, images_per_s=1 / sec_per_img, device=str(device), amp=not a.no_amp and device.type == "cuda")
    np.savez_compressed(out, images=np.concatenate(images), labels=labels.numpy(), meta=json.dumps(meta))
    print(json.dumps(meta))
    return meta


if __name__ == "__main__":
    main()
