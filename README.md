# dit-cifar10

Diffusion Transformer (DiT, Peebles and Xie 2023) implemented from scratch in PyTorch and trained class-conditionally on CIFAR-10, to test three of the paper's claims at small scale and to measure how far sampling can be cut before quality breaks.

Status: model, diffusion math, samplers, EMA and trainer are written and unit-tested on CPU. No training has run yet; there are no results.

## Questions

1. Does a transformer denoiser match a U-Net at this scale? Best DiT vs the pretrained `google/ddpm-cifar10-32` U-Net under the same sampler and step count.
2. Does FID track transformer Gflops? DiT-S/4, S/2 and B/4 trained with one recipe. B/4 has the same Gflops as S/2 with 4x the parameters.
3. How far can sampling be cut? DDPM 1000, DDIM 250/50/10/4, DPM-Solver++ 50/10/4: FID against seconds per image.
4. What does classifier-free guidance cost and buy? Scale sweep 1.0, 1.5, 2.0, 4.0: FID, Inception Score, class accuracy.

## Configurations

| Name | Depth | Width | Heads | Patch | Tokens | Params | Gflops |
|---|---|---|---|---|---|---|---|
| DiT-S/4 | 12 | 384 | 6 | 4 | 64 | 32.5M | 1.41 |
| DiT-S/2 | 12 | 384 | 6 | 2 | 256 | 32.5M | 6.05 |
| DiT-B/4 | 12 | 768 | 12 | 4 | 64 | 129.6M | 5.56 |

Counts from `dit.model.count_params` and `dit.model.forward_gflops`, within 3% and 10% of the paper's table (the paper's models take 4-channel latents; ours take 3-channel pixels).

## Layout

```
src/dit/model.py      DiT: patchify, 2-D sin-cos pos embed, adaLN-Zero blocks, zero-init final layer, CFG forward
src/dit/diffusion.py  linear beta schedule, q_sample, learn-sigma VLB loss, DDPM and DDIM samplers
src/dit/ema.py        EMA of trainable parameters
src/dit/train.py      AdamW, bf16 autocast, grad clip, checkpoint-resume, DDP via torchrun
src/dit/data.py       CIFAR-10 loader and a fake dataset for smoke tests
tests/                27 tests, CPU only
```

## Run

```
uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q
.venv/bin/python -m dit.train --model DiT-S/4 --steps 100000 --batch 128 --out checkpoints/s4
torchrun --nproc_per_node 2 -m dit.train --model DiT-S/2 --steps 100000 --batch 128 --out checkpoints/s2
```

## Recipe

Linear betas 1e-4 to 0.02 over 1000 steps; epsilon prediction with MSE plus the Improved-DDPM variational term on the learned variance (mean frozen); AdamW lr 1e-4, no weight decay; EMA 0.9999; label dropout 0.1; horizontal flip only; global batch 128; bf16 autocast with fp32 master weights; gradient clipping at 1.0. These are the DiT paper's settings except for batch size (paper: 256) and step count.

## References

Peebles and Xie, Scalable Diffusion Models with Transformers, ICCV 2023. Ho et al., DDPM, NeurIPS 2020. Nichol and Dhariwal, Improved DDPM, ICML 2021. Song et al., DDIM, ICLR 2021. Lu et al., DPM-Solver++, 2022.
