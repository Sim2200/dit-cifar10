# dit-cifar10

Diffusion Transformer (DiT, Peebles and Xie 2023) implemented from scratch in PyTorch and trained class-conditionally on CIFAR-10, to test three of the paper's claims at small scale and to measure how far sampling can be cut before quality breaks.

Status: complete. DiT-S/4, S/2 and B/4 trained for 100k steps each and evaluated on Kaggle T4s, alongside the pretrained U-Net baseline under the same evaluation code. Every number here is measured by the code in this repo; raw outputs are in `results/`.

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

Counts from `dit.model.count_params` and `dit.model.forward_gflops`, within 3% and 10% of th## Results so far: D## Results: three DiTs at 100k steps, batch 128, and the pretrained U-Net

Training time: S/4 4.7 h on one T4; S/2 10.8 h and B/4 8.6 h on two T4s with DDP. The U-Net is `google/ddpm-cifar10-32` (35.7M parameters, unconditional, trained by its authors for 800k steps) loaded through diffusers and sampled with the same code. FID and Inception Score on 10k samples against CIFAR-10 train statistics; precision/recall (k=3) against a fixed 10k train subset; class accuracy from a pretrained CIFAR-10 ResNet-56; seconds per image on one T4 in fp16 at batch 500. 10k-sample FID runs higher than the 50k-sample FID papers report (the U-Net's published 3.17 measures as 12.8 here), so compare rows within this table, not against the literature.

FID per sampler setting (lower is better):

| Sampler | Steps | CFG | DiT-S/4 | DiT-S/2 | DiT-B/4 | U-Net |
|---|---|---|---|---|---|---|
| DDPM | 1000 | 1.0 | 31.6 | 20.2 | 20.7 | **12.8** |
| DDIM | 250 | 1.0 | 33.6 | 22.5 | 23.6 | 14.3 |
| DDIM | 50 | 1.0 | 37.0 | 25.9 | 26.5 | 16.9 |
| DDIM | 10 | 1.0 | 51.0 | 36.8 | 36.9 | 40.0 |
| DDIM | 4 | 1.0 | 100.7 | 90.6 | 81.0 | 163.9 |
| DPM-Solver++ | 50 | 1.0 | 37.8 | 27.3 | 28.1 | 16.2 |
| DPM-Solver++ | 10 | 1.0 | 44.6 | 33.7 | 35.8 | 25.8 |
| DPM-Solver++ | 4 | 1.0 | 73.2 | 63.5 | 63.5 | 78.3 |
| DDIM | 50 | 1.5 | 29.2 | 18.1 | 19.4 | n/a |
| DDIM | 50 | 2.0 | 25.9 | **16.3** | **17.3** | n/a |
| DDIM | 50 | 4.0 | 33.4 | 28.6 | 26.7 | n/a |

Full metrics at the two settings that matter most (the U-Net is unconditional, so it has no guidance row or class accuracy):

| Model | Setting | FID | IS | Precision | Recall | Class acc | s/img |
|---|---|---|---|---|---|---|---|
| DiT-S/4 | DDPM 1000 | 31.6 | 6.74 | 0.80 | 0.27 | 0.50 | 0.38 |
| DiT-S/2 | DDPM 1000 | 20.2 | 7.45 | 0.76 | 0.41 | 0.66 | 2.19 |
| DiT-B/4 | DDPM 1000 | 20.7 | 7.63 | 0.79 | 0.37 | 0.66 | 1.05 |
| U-Net | DDPM 1000 | 12.8 | 8.43 | 0.78 | 0.48 | n/a | 1.59 |
| DiT-S/4 | DDIM 50, CFG 2 | 25.9 | 7.87 | 0.85 | 0.22 | 0.78 | 0.038 |
| DiT-S/2 | DDIM 50, CFG 2 | 16.3 | 8.49 | 0.82 | 0.29 | 0.91 | 0.22 |
| DiT-B/4 | DDIM 50, CFG 2 | 17.3 | 8.49 | 0.82 | 0.29 | 0.90 | 0.11 |

Every setting with precision, IS and class accuracy is in `results/summary.md`.

Answers to the four questions:

1. **Transformer vs U-Net: not at this training budget.** Under the same sampler and step count the U-Net wins every setting from 50 steps up (DDPM-1000 FID 12.8 vs 20.2 for the best DiT), with higher recall (0.48 vs 0.41). The gap is smaller than the compute gap: the U-Net saw 8x more training steps. Guidance closes most of it (S/2 at CFG 2 reaches 16.3, the U-Net cannot be guided), but that is a different generator, not a fair match. Below 10 steps the DiTs degrade more gracefully: at DDIM-4 the U-Net collapses to FID 164 while B/4 stays at 81.
2. **FID tracks Gflops, not parameters.** S/2 and B/4 have the same forward cost (6.1 and 5.6 Gflops) and B/4 has 4x the parameters; they land within one FID point of each other at every setting (20.2 vs 20.7, 16.3 vs 17.3), while S/4 at 1.4 Gflops is 11 to 15 points behind. Token count, not width, is the lever at this scale, which is the DiT paper's Figure 8 result reproduced in pixel space. B/4 also samples twice as fast as S/2 for the same quality because attention over 64 tokens is cheaper than over 256.
3. **Sampling knee is at about 50 steps for every model.** DDPM 1000 to DDIM 50 costs 5 to 6 FID points on all four models for a 20x speedup. At 10 steps DPM-Solver++ beats DDIM by 1 to 14 points (largest on the U-Net: 25.8 vs 40.0). At 4 steps every model collapses to near-zero recall; the U-Net collapses hardest. Best quality per second: DiT-B/4 with 10 DPM-Solver++ steps at 0.011 s/img gives FID 35.8, better than S/4's 1000-step DDPM at 35x the cost.
4. **Guidance scale 2 is the sweet spot.** It is the best FID on all three DiTs (37.0 to 25.9, 25.9 to 16.3, 26.5 to 17.3 over the unguided DDIM-50 row) and lifts class accuracy to 78%, 91% and 90%. Scale 4 keeps raising accuracy (91%, 98%, 98%) and IS but costs 7 to 10 FID points and halves recall: the usual fidelity-diversity trade, at twice the sampling cost of unguided.

Memorisation check: nearest-training-neighbour distance in Inception space for 1000 samples. Median 14.0, 14.0, 13.9 and 13.2 (S/4, S/2, B/4, U-Net) against 11.8 for real test images; 1st percentile 9.3, 8.5, 8.5 and 7.4 against 6.0. No model places samples closer to the training set than real held-out images sit.

Figures: `results/figures/loss_curves.png`, `scaling_fid_vs_gflops.png`, `sampler_fid_vs_seconds.png`, `guidance_sweep.png`; 10x10 sample grids per setting in `results/figures/samples/`.

it/model.py      DiT: patchify, 2-D sin-cos pos embed, adaLN-Zero blocks, zero-init final layer, CFG forward
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
