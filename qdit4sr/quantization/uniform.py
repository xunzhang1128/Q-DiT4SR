"""Online MAD normalization and the RobustUQ uniform quantizer."""
import math

import torch
from torch import nn


_TABLE = {
    1: (0.797885, 0.797885), 2: (1.991374, 0.995687),
    3: (2.344078, 0.586019), 4: (2.681605, 0.335201),
    5: (3.010220, 0.188139), 6: (3.330017, 0.104063),
    7: (3.639531, 0.056868), 8: (3.937585, 0.030762),
}


class UniformQuantSTE(nn.Module):
    def __init__(self, bits):
        super().__init__()
        if bits not in _TABLE:
            raise ValueError("RobustUQ bits must be an integer in [1, 8]")
        self.bits = bits
        a, delta = _TABLE[bits]
        self.qmin, self.qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        if bits > 1 and abs(a / delta - self.qmax) > 1e-3:
            delta = a / self.qmax
        self.register_buffer("a_opt", torch.tensor(a))
        self.register_buffer("delta", torch.tensor(delta))

    def forward(self, x):
        if self.bits == 1:
            reconstructed = torch.where(x >= 0, self.a_opt, -self.a_opt)
        else:
            codes = torch.round(torch.clamp(x, -self.a_opt, self.a_opt) / self.delta)
            reconstructed = codes.clamp(self.qmin, self.qmax) * self.delta
        # Preserve the original STE forward arithmetic, including FP16 rounding.
        return x + (reconstructed - x).detach()


class RobustUQ(nn.Module):
    def __init__(self, bits):
        super().__init__()
        self.quantizer = UniformQuantSTE(bits)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        std = (centered.abs().mean(dim=-1, keepdim=True) * math.sqrt(math.pi / 2)).clamp(min=1e-6)
        return self.quantizer(centered / std) * std + mean
