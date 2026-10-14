"""Exact block SVD with a parameter budget expressed as a global rank."""
from functools import lru_cache
from math import isqrt

import torch
from torch import nn
from torch.nn import functional as F


def _divisors(n):
    return sorted({d for i in range(1, isqrt(n) + 1) if n % i == 0 for d in (i, n // i) if d >= 2})


@lru_cache(maxsize=None)
def block_config(in_features, out_features, rank_budget):
    target = rank_budget * (in_features + out_features)
    if target <= 0:
        raise ValueError("BLR rank budget must be positive")
    best, error = None, float("inf")
    for bo in _divisors(out_features):
        for bi in _divisors(in_features):
            grid = (out_features // bo) * (in_features // bi)
            for rank in range(1, min(64, bo, bi) + 1):
                distance = abs(grid * rank * (bo + bi) - target)
                if distance < error:
                    best, error = (bo, bi, rank), distance
    if best is None or error / target > 0.1:
        raise ValueError("No BLR blocks fit the rank budget for (%d, %d)" % (out_features, in_features))
    return best


class BlockLowRank(nn.Linear):
    def __init__(self, weight, rank_budget):
        out_features, in_features = weight.shape
        # Retain the legacy initialization order for seeded random transforms.
        super().__init__(in_features, out_features, bias=False)
        self.bo, self.bi, self.rank = block_config(in_features, out_features, rank_budget)
        self.mo, self.mi = out_features // self.bo, in_features // self.bi
        self.U = nn.Parameter(weight.new_empty(self.mo, self.mi, self.bo, self.rank))
        self.V = nn.Parameter(weight.new_empty(self.mo, self.mi, self.rank, self.bi))
        blocks = weight.view(self.mo, self.bo, self.mi, self.bi).permute(0, 2, 1, 3)
        with torch.no_grad():
            for p in range(self.mo):
                for q in range(self.mi):
                    u, s, vh = torch.linalg.svd(blocks[p, q], full_matrices=False)
                    root = s[:self.rank].sqrt()
                    self.U[p, q].copy_(u[:, :self.rank] * root.unsqueeze(0))
                    self.V[p, q].copy_(root.unsqueeze(1) * vh[:self.rank])
        del self.weight

    def equivalent_weight(self):
        blocks = torch.einsum("pqkr,pqri->pqki", self.U, self.V)
        return blocks.permute(0, 2, 1, 3).contiguous().view(self.out_features, self.in_features)

    def forward(self, x):
        # Match the reference implementation, which reconstructs W before F.linear.
        return F.linear(x, self.equivalent_weight())
