import numpy as np
import torch

from dit.evaluate import Uint8Images, nearest_neighbour_distances


def test_uint8_dataset_layout():
    arr = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    ds = Uint8Images(arr)
    x = ds[0]
    assert len(ds) == 4 and x.shape == (3, 32, 32) and x.dtype == torch.uint8


def test_nearest_neighbour_distances_exact_match_is_zero():
    bank = torch.randn(50, 8)
    q = torch.cat([bank[:3], bank[:3] + 1.0])
    d = nearest_neighbour_distances(q, bank, torch.device("cpu"), chunk=4)
    assert torch.allclose(d[:3], torch.zeros(3), atol=1e-5)
    assert torch.all(d[3:] > 0) and torch.all(d[3:] <= np.sqrt(8) + 1e-5)
