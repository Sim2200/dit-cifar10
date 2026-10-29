"""Training loop: AdamW lr 1e-4, EMA 0.9999, bf16 autocast on CUDA, grad clip 1.0,
checkpoint-resume, optional DistributedDataParallel via torchrun.

    python -m dit.train --model DiT-S/4 --steps 100000 --out checkpoints/s4
    torchrun --nproc_per_node 2 -m dit.train --model DiT-B/2 ...
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .data import FakeImages, cifar10
from .diffusion import GaussianDiffusion
from .ema import EMA
from .model import build_dit, count_params


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="DiT-S/4")
    p.add_argument("--data", default="data")
    p.add_argument("--fake-data", action="store_true")
    p.add_argument("--out", default="checkpoints/run")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch", type=int, default=128, help="global batch size")
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ema", type=float, default=0.9999)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--ckpt-every", type=int, default=10_000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--keep", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-learn-sigma", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--resume", default=None, help="checkpoint path; default: latest in --out")
    return p.parse_args(argv)


def setup_dist():
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return rank, world, torch.device("cuda", rank % torch.cuda.device_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, device


def latest_checkpoint(out):
    cks = sorted(Path(out).glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return cks[-1] if cks else None


def save_checkpoint(out, step, model, ema, opt, args, keep):
    path = Path(out) / f"step_{step:07d}.pt"
    torch.save(dict(step=step, model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), args=vars(args)), path)
    cks = sorted(Path(out).glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    for old in cks[:-keep]:
        old.unlink()
    return path


def main(argv=None):
    args = parse_args(argv)
    rank, world, device = setup_dist()
    is_main = rank == 0
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    out = Path(args.out)
    if is_main:
        out.mkdir(parents=True, exist_ok=True)

    model = build_dit(args.model, learn_sigma=not args.no_learn_sigma).to(device)
    ema = EMA(model, decay=args.ema)
    diffusion = GaussianDiffusion(learn_sigma=not args.no_learn_sigma)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0, betas=(0.9, 0.999))

    step = 0
    ck = Path(args.resume) if args.resume else latest_checkpoint(out)
    if ck is not None and ck.exists():
        sd = torch.load(ck, map_location=device)
        model.load_state_dict(sd["model"])
        ema.load_state_dict(sd["ema"])
        opt.load_state_dict(sd["opt"])
        step = sd["step"]
        if is_main:
            print(f"resumed from {ck} at step {step}")

    ddp_model = DDP(model, device_ids=[device.index]) if world > 1 else model

    ds = FakeImages(n=256) if args.fake_data else cifar10(args.data, train=True, download=is_main)
    if world > 1:
        dist.barrier()
    assert args.batch % (world * args.accum) == 0
    micro = args.batch // (world * args.accum)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed) if world > 1 else None
    loader = DataLoader(ds, batch_size=micro, shuffle=sampler is None, sampler=sampler, num_workers=args.workers, drop_last=True, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)

    use_amp = device.type == "cuda" and not args.no_amp
    log_path = out / "train_log.csv"
    if is_main:
        print(f"{args.model}: {count_params(model)/1e6:.1f}M params, world={world}, micro-batch={micro}, amp={use_amp}")
        (out / "config.json").write_text(json.dumps(vars(args), indent=2))
        if not log_path.exists():
            log_path.write_text("step,loss,mse,vb,img_per_s,lr\n")

    model.train()
    epoch = 0
    t0 = time.time()
    seen = 0
    running = dict(loss=0.0, mse=0.0, vb=0.0, n=0)
    done = step >= args.steps
    while not done:
        if sampler is not None:
            sampler.set_epoch(epoch)
        it = iter(loader)
        while not done:
            opt.zero_grad(set_to_none=True)
            for a in range(args.accum):
                try:
                    x, y = next(it)
                except StopIteration:
                    break
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                t = torch.randint(0, diffusion.timesteps, (x.shape[0],), device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    terms = diffusion.training_losses(ddp_model, x, t, dict(y=y))
                loss = terms["loss"].mean() / args.accum
                loss.backward()
                running["loss"] += terms["loss"].mean().item()
                running["mse"] += terms["mse"].mean().item()
                running["vb"] += terms["vb"].mean().item() if "vb" in terms else 0.0
                running["n"] += 1
                seen += x.shape[0] * world
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()
                ema.update(model)
                step += 1
                if is_main and step % args.log_every == 0:
                    n = max(running["n"], 1)
                    ips = seen / (time.time() - t0)
                    with open(log_path, "a") as f:
                        csv.writer(f).writerow([step, running["loss"] / n, running["mse"] / n, running["vb"] / n, f"{ips:.1f}", args.lr])
                    print(f"step {step} loss {running['loss']/n:.4f} mse {running['mse']/n:.4f} vb {running['vb']/n:.4f} {ips:.1f} img/s")
                    running = dict(loss=0.0, mse=0.0, vb=0.0, n=0)
                    t0, seen = time.time(), 0
                if is_main and (step % args.ckpt_every == 0 or step == args.steps):
                    save_checkpoint(out, step, model, ema, opt, args, args.keep)
                if step >= args.steps:
                    done = True
                continue
            break
        epoch += 1

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
