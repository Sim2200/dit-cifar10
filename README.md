# dit-cifar10

Diffusion Transformer (DiT, Peebles and Xie 2023) implemented from scratch in PyTorch and trained class-conditionally on CIFAR-10, to test three of the paper's claims at small scale and to measure how far sampling can be cut before quality breaks.

Status: DiT-S/4 trained and evaluated on Kaggle T4s (below). DiT-S/2 and DiT-B/4 are in progress. Every number here is measured by the code in this repo; raw outputs are in `results/`.

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

## Results so far: DiT-S/4, 100k steps, batch 128, 4.7 h on one T4

FID and Inception Score on 10k samples against CIFAR-10 train statistics; precision/recall (k=3) against a fixed 10k train subset; class accuracy from a pretrained CIFAR-10 ResNet-56; seconds per image measured on one T4 in fp16 at batch 500.

| Sampler | Steps | CFG | FID | IS | Precision | Recall | Class acc | s/img |
|---|---|---|---|---|---|---|---|---|
| DDPM | 1000 | 1.0 | 31.6 | 6.74 | 0.80 | 0.27 | 0.50 | 0.381 |
| DDIM | 250 | 1.0 | 33.6 | 6.75 | 0.78 | 0.28 | 0.49 | 0.096 |
| DDIM | 50 | 1.0 | 37.0 | 6.62 | 0.82 | 0.22 | 0.46 | 0.019 |
| DDIM | 10 | 1.0 | 51.0 | 6.33 | 0.94 | 0.06 | 0.39 | 0.0040 |
| DDIM | 4 | 1.0 | 100.7 | 4.92 | 0.99 | 0.00 | 0.24 | 0.0016 |
| DPM-Solver++ | 50 | 1.0 | 37.8 | 6.65 | 0.78 | 0.27 | 0.47 | 0.019 |
| DPM-Solver++ | 10 | 1.0 | 44.6 | 6.56 | 0.88 | 0.11 | 0.40 | 0.0039 |
| DPM-Solver++ | 4 | 1.0 | 73.2 | 5.47 | 0.99 | 0.00 | 0.27 | 0.0016 |
| DDIM | 50 | 1.5 | 29.2 | 7.38 | 0.83 | 0.23 | 0.66 | 0.038 |
| DDIM | 50 | 2.0 | **25.9** | 7.87 | 0.85 | 0.22 | 0.78 | 0.038 |
| DDIM | 50 | 4.0 | 33.4 | 8.24 | 0.84 | 0.10 | **0.91** | 0.038 |

What the smallest model already shows:

- Sampling can be cut 20x (DDPM 1000 to DDIM 50) for 5 FID points; at 10 steps DPM-Solver++ beats DDIM by 6 FID points; at 4 steps both collapse to near-zero recall.
- Guidance at scale 2 is the best FID (25.9, from 37.0) and lifts class accuracy from 46% to 78%; scale 4 pushes accuracy to 91% at the cost of FID and recall, the usual fidelity-diversity trade.
- Memorisation check: nearest-training-neighbour distance in Inception space, median 14.0 for generated vs 11.8 for real test images (1st percentile 9.3 vs 6.0), so samples are not closer to the training set than real held-out images are.
- FID 31.6 is far from the U-Net's published 3.17: S/4 is 1.4 Gflops per forward and this run is about 3% of the paper's training compute. The scaling runs (S/2 at 4.3x the compute, B/4 at 4x) are what tests whether quality follows compute.

Figures: `results/figures/loss_curves.png`, `sampler_fid_vs_seconds.png`, `guidance_sweep.png`; sample grids per setting in `results/figures/samples/`.

## Layout

```
src/dit/model.py      DiT: patchify, 2-D sin-cos pos embed, adaLN-Zero blocks, zero-init final layer, CFG forward
src/dit/diffusion.py  linear beta schedule, q_sample, learn-sigma VLB loss, DDPM and DDIM samplers
src/dit/ema.py        EMA of trainable parameters
src/dit/train.py      AdamW, bf16 autocast, grad clip, checkpoint-resume, DDP via torchrun
src/dit/data.py       CIFAR-10 loader and a fake dataset for smoke tests
kaggle/run_pilot.py   throughput pilot on 2x T4: img/s and peak memory per config and precision, sampler timing
scripts/run_on_kaggle.py  pushes src as a Kaggle dataset, runs a kaggle/ script as a GPU kernel, pulls results/
tests/                27 tests, CPU only
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
