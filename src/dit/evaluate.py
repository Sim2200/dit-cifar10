"""Metrics for a sample set against the CIFAR-10 training set.

    python -m dit.evaluate samples/s4_ddim50.npz --data data --out results/eval_s4_ddim50.json

FID, Inception Score, precision/recall (k=3) via torch-fidelity on Inception-v3 features, with
CIFAR-10 train statistics cached under --cache. Class accuracy via a pretrained CIFAR-10
ResNet-56 (chenyaofo/pytorch-cifar-models, torch.hub). Memorisation: nearest training neighbour
in Inception feature space for 1,000 samples, compared with the same statistic for real test images.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class Uint8Images(Dataset):
    def __init__(self, arr):
        self.arr = arr  # N, H, W, 3 uint8

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, i):
        return torch.from_numpy(self.arr[i]).permute(2, 0, 1)


def cifar_uint8(root, train=True):
    from torchvision import datasets

    ds = datasets.CIFAR10(root, train=train, download=True)
    return ds.data, np.asarray(ds.targets)


def fidelity_metrics(samples, real, cache_dir, cache_name="cifar10-train", kid=False):
    import torch_fidelity

    return torch_fidelity.calculate_metrics(
        input1=Uint8Images(samples), input2=Uint8Images(real), cuda=torch.cuda.is_available(),
        isc=True, fid=True, kid=kid, prc=True, prc_neighborhood=3,
        input2_cache_name=cache_name, cache_root=str(cache_dir), verbose=False,
    )


def load_classifier(device):
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True, verbose=False)
    return model.to(device).eval()


@torch.no_grad()
def class_accuracy(samples, labels, device, batch=500):
    clf = load_classifier(device)
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)
    correct, per_class = 0, np.zeros((10, 2))
    for i in range(0, len(samples), batch):
        x = torch.from_numpy(samples[i : i + batch]).permute(0, 3, 1, 2).float().div(255).to(device)
        pred = clf((x - mean) / std).argmax(1).cpu().numpy()
        y = labels[i : i + batch]
        correct += (pred == y).sum()
        for c in range(10):
            m = y == c
            per_class[c] += [(pred[m] == c).sum(), m.sum()]
    return dict(accuracy=float(correct / len(samples)), per_class=(per_class[:, 0] / np.maximum(per_class[:, 1], 1)).round(4).tolist())


@torch.no_grad()
def inception_features(images, device, batch=250):
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

    fe = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).to(device).eval()
    feats = []
    for i in range(0, len(images), batch):
        x = torch.from_numpy(images[i : i + batch]).permute(0, 3, 1, 2).to(device)
        feats.append(fe(x)[0].cpu())
    return torch.cat(feats)


@torch.no_grad()
def nearest_neighbour_distances(query, bank, device, chunk=1000):
    bank = bank.to(device)
    out = []
    for i in range(0, len(query), chunk):
        q = query[i : i + chunk].to(device)
        out.append(torch.cdist(q, bank, compute_mode="donot_use_mm_for_euclid_dist").min(dim=1).values.cpu())
    return torch.cat(out)


def memorisation(samples, train_feats, test_feats, device, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(samples), min(n, len(samples)), replace=False)
    d_gen = nearest_neighbour_distances(inception_features(samples[idx], device), train_feats, device)
    d_real = nearest_neighbour_distances(test_feats[:n], train_feats, device)
    q = lambda d: dict(median=float(d.median()), p05=float(d.quantile(0.05)), p01=float(d.quantile(0.01)), min=float(d.min()))
    return dict(generated=q(d_gen), real_test=q(d_real), n=int(len(idx)))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("samples")
    p.add_argument("--data", default="data")
    p.add_argument("--cache", default="cache")
    p.add_argument("--out", required=True)
    p.add_argument("--no-memorisation", action="store_true")
    a = p.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(a.samples)
    samples, labels, meta = z["images"], z["labels"], json.loads(str(z["meta"]))
    train, _ = cifar_uint8(a.data, train=True)
    Path(a.cache).mkdir(parents=True, exist_ok=True)

    m = fidelity_metrics(samples, train, a.cache)
    result = dict(meta=meta, fid=m["frechet_inception_distance"], inception_score=m["inception_score_mean"], inception_score_std=m["inception_score_std"], precision=m["precision"], recall=m["recall"], f_score=m["f_score"])
    if meta.get("model") != "ddpm-cifar10-32-unet":
        result["class_accuracy"] = class_accuracy(samples, labels, device)
    if not a.no_memorisation:
        feats_path = Path(a.cache) / "inception_train_feats.pt"
        if feats_path.exists():
            train_feats, test_feats = torch.load(feats_path)
        else:
            test, _ = cifar_uint8(a.data, train=False)
            train_feats, test_feats = inception_features(train, device), inception_features(test, device)
            torch.save((train_feats, test_feats), feats_path)
        result["memorisation"] = memorisation(samples, train_feats, test_feats, device)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "meta"}, indent=1))
    return result


if __name__ == "__main__":
    main()
