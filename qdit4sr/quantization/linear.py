"""Signed Hadamard -> global SVD -> block SVD -> mixed W and scheduled A."""
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from .hadamard import transform
from .uniform import RobustUQ
from .blr import BlockLowRank


def target_linears(model):
    return {name: module for name, module in model.named_modules()
            if name.startswith("transformer_blocks.") and isinstance(module, nn.Linear)}


class GlobalLowRank(nn.Module):
    def __init__(self, weight, rank):
        super().__init__()
        out_features, in_features = weight.shape
        self.rank = min(rank, in_features, out_features)
        self.B = nn.Linear(in_features, self.rank, bias=False)
        self.A = nn.Linear(self.rank, out_features, bias=False)
        with torch.no_grad():
            u, s, v = torch.svd(weight)
            root = torch.diag(s[:self.rank].sqrt())
            self.A.weight = nn.Parameter(u[:, :self.rank] @ root)
            self.B.weight = nn.Parameter(root @ v[:, :self.rank].t())

    def equivalent_weight(self):
        return self.A.weight @ self.B.weight

    def forward(self, x):
        return self.A(self.B(x))


class MixedLinear(nn.Linear):
    def __init__(self, linear, w_bits, a_bits, svd_rank, blr_rank_budget, device):
        super().__init__(linear.in_features, linear.out_features, bias=linear.bias is not None)
        self.w_bits, self.a_bits = w_bits, a_bits
        weight = linear.weight.detach().to(device=device, dtype=torch.float32)
        signs = torch.randint(0, 2, (self.in_features,)).float() * 2 - 1
        self.register_buffer("signs", signs.to(device))
        weight = transform(weight * self.signs)
        self.variance = weight.var(dim=-1, unbiased=False).mean().item()
        self.svd = GlobalLowRank(weight, svd_rank) if svd_rank else None
        if self.svd is not None:
            weight = weight - self.svd.equivalent_weight().detach()
        self.blr = BlockLowRank(weight, blr_rank_budget) if blr_rank_budget else None
        if self.blr is not None:
            weight = weight - self.blr.equivalent_weight().detach()
        self.weight = nn.Parameter(weight)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().to(device=device, dtype=torch.float32))
        self.weight_quantizer = RobustUQ(w_bits)
        self.activation_quantizer = RobustUQ(a_bits)
        self.post_hadamard_tap = None
        self.to(device=device, dtype=torch.float16).requires_grad_(False)

    def forward(self, x):
        x = transform(x * self.signs)
        if self.post_hadamard_tap is not None:
            self.post_hadamard_tap(x)
        svd = self.svd(x) if self.svd is not None else None
        blr = self.blr(x) if self.blr is not None else None
        out = F.linear(self.activation_quantizer(x), self.weight_quantizer(self.weight), self.bias)
        if svd is not None:
            out = out + svd
        if blr is not None:
            out = out + blr
        return out


@torch.no_grad()
def quantize(model, bit_rows, config, device):
    layers = target_linears(model)
    rows = {row["layer_name"]: row for row in bit_rows}
    missing, extra = set(layers) - set(rows), set(rows) - set(layers)
    if missing or extra:
        raise ValueError("Bit map does not match checkpoint: missing=%s extra=%s" % (sorted(missing), sorted(extra)))
    for name, layer in layers.items():
        if rows[name]["n_params"] != layer.weight.numel():
            raise ValueError("Bit map shape mismatch for " + name)
    statistics, report = [], []
    ordered_layers = [(name, layers[name]) for name in rows]
    for name, layer in tqdm(ordered_layers, total=len(layers), desc="Quantizing", mininterval=2):
        q = MixedLinear(layer, rows[name]["bits_assigned"], config["a_bits"],
                        config["svd_rank"], config["blr_rank_budget"], device)
        parent_name, leaf = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), leaf, q)
        statistics.append(dict(layer_name=name, variance=q.variance, n_params=q.weight.numel()))
        entry = dict(layer_name=name, in_features=q.in_features, out_features=q.out_features,
                     w_bits=q.w_bits, a_bits=q.a_bits,
                     svd_rank=q.svd.rank if q.svd else 0,
                     svd_parameters=sum(p.numel() for p in q.svd.parameters()) if q.svd else 0,
                     blr_parameters=sum(p.numel() for p in q.blr.parameters()) if q.blr else 0)
        if q.blr is not None:
            entry.update(blr_block=[q.blr.bo, q.blr.bi], blr_grid=[q.blr.mo, q.blr.mi], blr_block_rank=q.blr.rank)
        report.append(entry)
    model.to(device=device, dtype=torch.float16).eval().requires_grad_(False)
    return statistics, report
