import torch
from torch.utils.data import Dataset


def cifar10(root, train=True, download=True):
    from torchvision import datasets, transforms

    tf = transforms.Compose(
        [transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
    )
    return datasets.CIFAR10(root, train=train, download=download, transform=tf)


class FakeImages(Dataset):
    """Random images in [-1, 1] with labels, for smoke tests without a download."""

    def __init__(self, n=64, num_classes=10, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.rand(n, 3, 32, 32, generator=g) * 2 - 1
        self.y = torch.randint(0, num_classes, (n,), generator=g)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]
