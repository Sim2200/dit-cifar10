"""Result figures from results/*.json and results/train_log_*.csv into results/figures/.

    python scripts/plots.py

Each figure is drawn only if its inputs exist, so the script can run at any stage.
Categorical colours follow a fixed model order (S/4, S/2, B/4, U-Net); one y-axis per chart.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

ORDER = ["s4", "s2", "b4", "unet"]
LABEL = {"s4": "DiT-S/4", "s2": "DiT-S/2", "b4": "DiT-B/4", "unet": "DDPM U-Net"}
COLOR = {"s4": "#2a78d6", "s2": "#eb6834", "b4": "#1baf7a", "unet": "#eda100"}
GFLOPS = {"s4": 1.41, "s2": 6.05, "b4": 5.56, "unet": 6.1}
MARKER = {"ddpm": "o", "ddim": "s", "dpmpp": "^"}
SAMPLER = {"ddpm": "DDPM", "ddim": "DDIM", "dpmpp": "DPM-Solver++"}

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8a85", "axes.facecolor": "#fcfcfb", "figure.facecolor": "#fcfcfb", "legend.frameon": False})


def load_evals():
    out = {}
    for f in RES.glob("eval_*.json"):
        d = json.loads(f.read_text())
        if "results" in d:
            out[d["config"]["run"] if not d["config"]["unet"] else "unet"] = d["results"]
    return out


def loss_curves():
    logs = {r: RES / f"train_log_{r}.csv" for r in ("s4", "s2", "b4")}
    logs = {r: p for r, p in logs.items() if p.exists()}
    if not logs:
        return
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for r, p in logs.items():
        rows = list(csv.DictReader(open(p)))
        steps = [int(x["step"]) for x in rows]
        mse = [float(x["mse"]) for x in rows]
        k = 10
        sm = [sum(mse[max(0, i - k + 1) : i + 1]) / len(mse[max(0, i - k + 1) : i + 1]) for i in range(len(mse))]
        ax.plot(steps, sm, color=COLOR[r], lw=2, label=LABEL[r])
        ax.annotate(f"{sm[-1]:.4f}", (steps[-1], sm[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color="#444")
    ax.set_xlabel("training step")
    ax.set_ylabel("epsilon MSE (1k-step mean, log)")
    ax.set_title("Training loss, batch 128, fp16 on T4", loc="left")
    ax.set_yscale("log")
    if len(logs) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "loss_curves.png")
    plt.close(fig)


def pick(results, sampler, steps, cfg=1.0):
    for r in results:
        m = r["meta"]
        if m["sampler"] == sampler and m["steps"] == steps and m.get("cfg", 1.0) == cfg:
            return r
    return None


def scaling(evals):
    pts = [(r, pick(evals[r], "ddim", 50)) for r in ("s4", "s2", "b4") if r in evals]
    pts = [(r, e) for r, e in pts if e]
    if len(pts) < 2:
        return
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for r, e in pts:
        ax.scatter(GFLOPS[r], e["fid"], color=COLOR[r], s=60, zorder=3)
        ax.annotate(f"{LABEL[r]}  FID {e['fid']:.1f}", (GFLOPS[r], e["fid"]), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("forward Gflops (log)")
    ax.set_ylabel("FID, 10k samples, DDIM 50")
    ax.set_title("Sample quality vs transformer compute, 100k steps", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "scaling_fid_vs_gflops.png")
    plt.close(fig)


def sampler_sweep(evals):
    runs = [r for r in ORDER if r in evals]
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for r in runs:
        for s in ("ddpm", "ddim", "dpmpp"):
            es = sorted([e for e in evals[r] if e["meta"]["sampler"] == s and e["meta"].get("cfg", 1.0) == 1.0], key=lambda e: e["meta"]["seconds_per_image"])
            if not es:
                continue
            x = [e["meta"]["seconds_per_image"] for e in es]
            y = [e["fid"] for e in es]
            ax.plot(x, y, color=COLOR[r], marker=MARKER[s], ms=6, lw=1.5, mec="#fcfcfb", mew=1, label=f"{LABEL[r]}, {SAMPLER[s]}")
            for e in es:
                ax.annotate(str(e["meta"]["steps"]), (e["meta"]["seconds_per_image"], e["fid"]), xytext=(4, 3), textcoords="offset points", fontsize=7, color="#555")
    ax.set_xscale("log")
    ax.set_xlabel("seconds per image on one T4 (log)")
    ax.set_ylabel("FID, 10k samples")
    ax.set_title("Sampling cost vs quality; point labels are step counts", loc="left")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "sampler_fid_vs_seconds.png")
    plt.close(fig)


def guidance(evals):
    panels = []
    for r in ("s4", "s2", "b4"):
        if r not in evals:
            continue
        es = sorted([e for e in evals[r] if e["meta"]["sampler"] == "ddim" and e["meta"]["steps"] == 50 and "class_accuracy" in e], key=lambda e: e["meta"]["cfg"])
        if len(es) >= 2:
            panels.append((r, es))
    if not panels:
        return
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for r, es in panels:
        cfg = [e["meta"]["cfg"] for e in es]
        for ax, key, name in zip(axes, ("fid", "inception_score", "acc"), ("FID", "Inception Score", "class accuracy")):
            y = [e["class_accuracy"]["accuracy"] if key == "acc" else e[key] for e in es]
            ax.plot(cfg, y, color=COLOR[r], marker="o", ms=6, lw=2, mec="#fcfcfb", mew=1, label=LABEL[r])
            ax.set_xlabel("guidance scale")
            ax.set_title(name, loc="left")
            ax.set_xticks(cfg)
    if len(panels) > 1:
        axes[0].legend(fontsize=8)
    fig.suptitle("Classifier-free guidance sweep, DDIM 50, 10k samples", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "guidance_sweep.png")
    plt.close(fig)


def summary_table(evals):
    rows = ["| Model | Sampler | Steps | CFG | FID | IS | Precision | Recall | Class acc | s/img |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in ORDER:
        for e in sorted(evals.get(r, []), key=lambda e: (e["meta"]["sampler"], -e["meta"]["steps"], e["meta"].get("cfg", 1))):
            m = e["meta"]
            acc = f"{e['class_accuracy']['accuracy']:.3f}" if "class_accuracy" in e else "-"
            rows.append(f"| {LABEL[r]} | {SAMPLER[m['sampler']]} | {m['steps']} | {m.get('cfg', 1.0)} | {e['fid']:.2f} | {e['inception_score']:.2f} | {e['precision']:.3f} | {e['recall']:.3f} | {acc} | {m['seconds_per_image']:.4f} |")
    (RES / "summary.md").write_text("\n".join(rows) + "\n")


def main():
    loss_curves()
    evals = load_evals()
    if evals:
        scaling(evals)
        sampler_sweep(evals)
        guidance(evals)
        summary_table(evals)
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
