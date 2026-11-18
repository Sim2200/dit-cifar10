# dit-cifar10

Diffusion Transformer (DiT, Peebles and Xie 2023) implemented from scratch in PyTorch and trained class-conditionally on CIFAR-10, to test three of the paper's claims at small scale and to measure how far sampling can be cut before quality breaks.

Status: DiT-S/4 and DiT-S/2 trained and evaluated on Kaggle T4s (below). DiT-B/4 is trained (100k steps, 8.6 h on 2x T4) and its evaluation plus the pretrained U-Net baseline are running. Every number here is measured by the code in this repo; raw outputs are in `results/`.

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

Counts from `dit.model.count_params` and `dit.model.forward_gflops`, within 3% and 10% of th## Results so far: DiT-S/4 and DiT-S/2, 100k steps, batch 128

Training time: S/4 4.7 h on one T4; S/2 10.8 h on two T4s with DDP. FID and Inception Score on 10k samples against CIFAR-10 train statistics; precision/recall (k=3) against a fixed 10k train subset; class accuracy from a pretrained CIFAR-10 ResNet-56; seconds per image measured on one T4 in fp16 at batch 500.

| Sampler | Steps | CFG | S/4 FID | S/2 FID | S/4 IS | S/2 IS | S/4 recall | S/2 recall | S/4 acc | S/2 acc | S/4 s/img | S/2 s/img |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DDPM | 1000 | 1.0 | 31.6 | 20.2 | 6.74 | 7.45 | 0.27 | 0.41 | 0.50 | 0.66 | 0.381 | 2.19 |
| DDIM | 250 | 1.0 | 33.6 | 22.5 | 6.75 | 7.52 | 0.28 | 0.41 | 0.49 | 0.65 | 0.096 | 0.547 |
| DDIM | 50 | 1.0 | 37.0 | 25.9 | 6.62 | 7.35 | 0.22 | 0.34 | 0.46 | 0.63 | 0.019 | 0.110 |
| DDIM | 10 | 1.0 | 51.0 | 36.8 | 6.33 | 6.85 | 0.06 | 0.08 | 0.39 | 0.56 | 0.0040 | 0.022 |
| DDIM | 4 | 1.0 | 100.7 | 90.6 | 4.92 | 5.38 | 0.00 | 0.00 | 0.24 | 0.30 | 0.0016 | 0.0089 |
| DPM-Solver++ | 50 | 1.0 | 37.8 | 27.3 | 6.65 | 7.37 | 0.27 | 0.39 | 0.47 | 0.63 | 0.019 | 0.109 |
| DPM-Solver++ | 10 | 1.0 | 44.6 | 33.7 | 6.56 | 7.20 | 0.11 | 0.17 | 0.40 | 0.58 | 0.0039 | 0.022 |
| DPM-Solver++ | 4 | 1.0 | 73.2 | 63.5 | 5.47 | 5.98 | 0.00 | 0.01 | 0.27 | 0.37 | 0.0016 | 0.0089 |
| DDIM | 50 | 1.5 | 29.2 | 18.1 | 7.38 | 8.08 | 0.23 | 0.33 | 0.66 | 0.83 | 0.038 | 0.222 |
| DDIM | 50 | 2.0 | **25.9** | **16.3** | 7.87 | 8.49 | 0.22 | 0.29 | 0.78 | 0.91 | 0.038 | 0.222 |
| DDIM | 50 | 4.0 | 33.4 | 28.6 | 8.24 | 8.35 | 0.10 | 0.15 | **0.91** | **0.98** | 0.038 | 0.221 |

Precision is 0.74 to 0.85 for every setting above 10 steps and 0.93 to 0.99 at 4 to 10 steps for both models; the full per-setting table with precision is `results/summary.md`.

What two models already show:

- Compute scaling (question 2, partial): going from 64 to 256 tokens at the same 32.5M parameters (1.4 to 6.1 Gflops) cuts FID by a third at every sampler setting: DDPM 31.6 to 20.2, DDIM-50 37.0 to 25.9, best guided 25.9 to 16.3. Recall rises from 0.27 to 0.41, so the gain is diversity as much as fidelity. B/4 (5.6 Gflops, 4x the parameters) will show whether Gflops or parameters is the lever.
- Sampling can be cut 20x (DDPM 1000 to DDIM 50) for about 5 FID points on both models; at 10 steps DPM-Solver++ beats DDIM by 3 to 6 FID points; at 4 steps both collapse to near-zero recall. S/2 at 10 DPM-Solver++ steps (FID 33.7, 0.022 s/img) is better and 17x faster than S/4 with 1000 DDPM steps.
- Guidance at scale 2 is the best FID on both models and lifts class accuracy from 46% to 78% (S/4) and 63% to 91% (S/2); scale 4 pushes accuracy to 91% and 98% at the cost of FID and recall, the usual fidelity-diversity trade.
- Memorisation check: nearest-training-neighbour distance in Inception space, median 14.0 for generated vs 11.8 for real test images for both models (1st percentile 9.3 and 8.5 vs 6.0), so samples are not closer to the training set than real held-out images are.
- FID 16 to 20 is still far from the U-Net's published 3.17: S/2 at 100k steps of batch 128 is about 12% of the paper's training compute for the model at this size, and the U-Net was trained for 800k steps.

Figures: `results/figures/loss_curves.png`, `scaling_fid_vs_gflops.png`, `sampler_fid_vs_seconds.png`, `guidance_sweep.png`; sample grids per setting in `results/figures/samples/`.

onds.png`, `guidance_sweep.png`; sample grids per setting in `results/figures/samples/`.

## Layout

```
src/dit/model.py      DiT: patchify, 2-D sin-cos pos embed, adaLN-Zero blocks, zero-init final layer, CFG forward
src/dit/diffusion.py  linear beta schedule, q_sample, learn-sigma VLB loss, DDPM and DDIM samplers
src/dit/ema.py        EMA of trainable parameters
src/dit/train.py      AdamW, bf16 autocast, grad clip, checkpoint-resume, DDP via torchrun
src/dit/data.py       CIFAR-10 loader and a fake dataset for smoke tests
kaggle/run_pilot.py   throughput pilot on 2x T4: img/s and peak memory per config and precision, sampler timing
scripts/run_on_kaggle.py  pushes src as a Kaggle dataset, runs a kaggle/ script as a GPU kernel, pulls results/
tests/                36 tests, CPU only
```

Precision: `--precision auto` uses bf16 where the GPU supports it (A100) and fp16 with a gradient scaler otherwise (T4).

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
