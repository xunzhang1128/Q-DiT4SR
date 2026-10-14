"""The power-of-two and 12 * power-of-two transforms used by SD3.5."""
from functools import lru_cache

import torch

try:
    from fast_hadamard_transform import hadamard_transform as _fast_transform
except ImportError:
    _fast_transform = None


@lru_cache(maxsize=1)
def _had12():
    return torch.tensor([
        [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1],
        [1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1],
        [1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1],
        [1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1],
        [1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1],
        [1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1],
        [1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1],
        [1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1],
        [1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1],
        [1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1],
        [1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1],
    ], dtype=torch.float32)


def _power_of_two(n):
    return n > 0 and n & (n - 1) == 0


def transform(x):
    n = x.shape[-1]
    k = 1 if _power_of_two(n) else 12
    if n % k or not _power_of_two(n // k):
        raise ValueError("Hadamard width must be 2^m or 12 * 2^m, got %d" % n)
    shape = x.shape
    y = x.reshape(-1, k, n // k).contiguous()
    scale = (1.0 / torch.tensor(n).sqrt()).item()
    if x.is_cuda:
        if _fast_transform is None:
            raise ImportError("CUDA quantization requires fast-hadamard-transform==1.0.4.post1")
        y = _fast_transform(y, scale=scale)
    else:
        stride = 1
        while stride < n // k:
            z = y.reshape(*y.shape[:-1], -1, 2, stride)
            left, right = z[..., 0, :], z[..., 1, :]
            y = torch.stack((left + right, left - right), dim=-2).reshape(y.shape)
            stride *= 2
        y = y * scale
    if k == 12:
        y = _had12().to(device=y.device, dtype=y.dtype) @ y
    return y.reshape(shape)
