"""Timestep DP with runtime RobustUQ distortion and per-image activation scheduling."""
import csv
import math
from collections import Counter, defaultdict

import numpy as np
import torch

from ..io import open_text, write_csv, write_json
from .linear import MixedLinear
from .uniform import RobustUQ


def load_schedule(path, steps=None):
    if path is None:
        return {}
    schedule = defaultdict(list)
    with open_text(path) as f:
        for row in csv.DictReader(f):
            start, end, bit = (int(row[k]) for k in ("step_start", "step_end", "bit"))
            if start < 0 or end < start or not 1 <= bit <= 8:
                raise ValueError("Invalid activation schedule row: " + str(row))
            schedule[row["layer_name"]].append((start, end, bit))
    if not schedule:
        raise ValueError("Empty activation schedule: " + str(path))
    for name, segments in schedule.items():
        segments.sort()
        if segments[0][0] != 0 or any(a[1] + 1 != b[0] for a, b in zip(segments, segments[1:])):
            raise ValueError("Activation schedule must cover consecutive steps for " + name)
        if steps is not None and segments[-1][1] < steps - 1:
            raise ValueError("Activation schedule does not cover all diffusion steps for " + name)
    return dict(schedule)


def estimate_kappa():
    """Gaussian relative MSE using the runtime normalization, clipping and quantizer."""
    z = torch.from_numpy(np.random.default_rng(123).standard_normal(200000).astype(np.float32))
    variance = z.var(unbiased=False)
    with torch.no_grad():
        return {bit: float((z - RobustUQ(bit)(z)).square().mean() / variance) for bit in range(2, 9)}


def solve_schedule(variance, steps, kappa, target=4.0):
    """Original segment DP; vectorize budgets while preserving strict tie order."""
    total, bits = len(variance), sorted(kappa)
    budget = int(math.floor(target * total))
    if not total or budget < min(bits) * total:
        raise ValueError("Infeasible activation bit budget")
    prefix = np.zeros(total + 1, dtype=np.float64)
    prefix[1:] = np.cumsum(variance.astype(np.float64))
    inf = 1e100
    shape = (total + 1, budget + 1, len(bits))
    dp = np.full(shape, inf, dtype=np.float64)
    prev_i, prev_c, prev_m = [np.full(shape, -1, dtype=np.int32) for _ in range(3)]
    dp[0, 0, :] = 0
    for j in range(1, total + 1):
        for i in range(j):
            seg_sum = float(prefix[j] - prefix[i])
            for m, bit in enumerate(bits):
                cost = bit * (j - i)
                if cost > budget:
                    continue
                distortion = kappa[bit] * seg_sum
                if i == 0:
                    if distortion < dp[j, cost, m]:
                        dp[j, cost, m] = distortion
                        prev_i[j, cost, m], prev_c[j, cost, m], prev_m[j, cost, m] = 0, 0, m
                    continue
                values = dp[i, :budget + 1 - cost, :]
                last = np.argmin(values, axis=1)
                best = values[np.arange(len(last)), last]
                candidate = best + distortion
                update = (best < inf / 10) & (candidate < dp[j, cost:, m])
                dp[j, cost:, m][update] = candidate[update]
                prev_i[j, cost:, m][update] = i
                prev_c[j, cost:, m][update] = np.arange(len(last))[update]
                prev_m[j, cost:, m][update] = last[update]
    c, m = np.unravel_index(np.argmin(dp[total]), dp[total].shape)
    best, j, segments = float(dp[total, c, m]), total, []
    if best >= inf / 10:
        raise ValueError("No feasible activation schedule")
    while j > 0:
        i, c0, m0 = int(prev_i[j, c, m]), int(prev_c[j, c, m]), int(prev_m[j, c, m])
        segments.append((int(steps[i]), int(steps[j - 1]), bits[m]))
        j, c, m = i, c0, m0
    segments.reverse()
    achieved = sum((end - start + 1) * bit for start, end, bit in segments) / total
    return segments, best, achieved, budget


def allocate_activation(statistics, output, target=4.0):
    import pandas as pd
    from pathlib import Path
    frame = pd.read_csv(statistics)
    frame = frame[frame["mean_token_var"].astype(float) != 0.0]
    kappa, rows = estimate_kappa(), []
    layers = sorted(frame["layer_name"].unique())
    if not layers:
        raise ValueError("No nonzero activation statistics")
    for index, name in enumerate(layers):
        series = frame[frame["layer_name"] == name].groupby("step_idx")["mean_token_var"].mean()
        steps, variance = series.index.to_numpy(np.int64), series.to_numpy(np.float64)
        order = np.argsort(steps)
        segments, distortion, achieved, budget = solve_schedule(variance[order], steps[order], kappa, target)
        for group, (start, end, bit) in enumerate(segments):
            rows.append(dict(layer_name=name, T=len(steps), target_avg_bit=target,
                             achieved_avg_bit=achieved, budget_B=budget, group_id=group,
                             step_start=start, step_end=end, bit=bit, total_distortion=distortion))
        if index % 50 == 0 or index + 1 == len(layers):
            print("[Schedule] %d/%d %s" % (index + 1, len(layers), name), flush=True)
    output = Path(output)
    write_csv(output / "activation_schedule.csv", rows)
    write_json(output / "estimated_kappa.json", kappa)
    return dict(layers=len(layers), segments=len(rows), target_avg_bit=target, kappa=kappa)


class ActivationBitScheduler:
    """Apply each layer's timestep schedule independently for every image."""
    def __init__(self, schedule, default_bits=4):
        self.schedule, self.default_bits = schedule, default_bits
        self.step_idx, self.last_t, self.handles = -1, None, []
        self.layers, self.bit_counts = [], Counter()

    def start_image(self):
        self.step_idx, self.last_t = -1, None
        self.bit_counts.clear()
        for module in self.layers:
            module._activation_segment_index = 0
            module._activation_last_step, module._activation_last_bit = -1, None

    def _update_step(self, timestep):
        value = int(timestep.flatten()[0].item()) if torch.is_tensor(timestep) else int(timestep)
        if self.last_t is None or value != self.last_t:
            self.step_idx += 1
            self.last_t = value

    def _get_bit_for_step(self, module, step):
        segments = module._activation_segments
        if not segments:
            return self.default_bits
        index = min(max(module._activation_segment_index, 0), len(segments) - 1)
        if step < segments[0][0]:
            module._activation_segment_index = 0
            return segments[0][2]
        while index < len(segments) and step > segments[index][1]:
            index += 1
        if index == len(segments):
            module._activation_segment_index = index - 1
            return segments[-1][2]
        module._activation_segment_index = index
        start, end, bit = segments[index]
        return bit if start <= step <= end else self.default_bits

    def attach(self, transformer):
        layers = {n: m for n, m in transformer.named_modules() if isinstance(m, MixedLinear)}
        if set(self.schedule) - set(layers):
            raise ValueError("Activation schedule contains unknown quantized layers")
        nonuniform = sum(len({b for _, _, b in s}) > 1 for s in self.schedule.values())
        self.layers = list(layers.values())
        def transformer_hook(module, args, kwargs):
            timestep = kwargs.get("timestep")
            if timestep is None and len(args) >= 3:
                timestep = args[2]
            if timestep is not None:
                self._update_step(timestep)
        self.handles.append(transformer.register_forward_pre_hook(transformer_hook, with_kwargs=True))
        for name, module in layers.items():
            module._activation_segments = self.schedule.get(name, [])
            bits = {self.default_bits} | {b for _, _, b in module._activation_segments}
            module._activation_quantizers = {b: RobustUQ(b).to(module.weight.device) for b in bits}
            def layer_hook(mod, inputs):
                if self.step_idx < 0 or mod._activation_last_step == self.step_idx:
                    return
                bit = self._get_bit_for_step(mod, self.step_idx)
                if bit != mod._activation_last_bit:
                    mod.activation_quantizer = mod._activation_quantizers[bit]
                    mod._activation_last_bit = bit
                mod._activation_last_step = self.step_idx
                self.bit_counts["w%da%d" % (mod.w_bits, mod.activation_quantizer.quantizer.bits)] += 1
            self.handles.append(module.register_forward_pre_hook(layer_hook))
        self.start_image()
        return dict(scheduled_layers=len(self.schedule), nonuniform_layers=nonuniform,
                    uniform_layers=len(self.schedule) - nonuniform, reset_between_images=True,
                    observation_unit="executed quantized layer-step; repeated tiles counted once")

    def detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.layers = []


def summarize_schedule(schedule, weights, steps=40, output=None):
    """Planned W/A counts, using the original layer-step and parameter-step units."""
    weights = {r["layer_name"]: r for r in weights}
    slots, pairs, param_pairs, seg_pairs = [], Counter(), Counter(), Counter()
    for name, segments in schedule.items():
        w, count = weights[name]["bits_assigned"], weights[name]["n_params"]
        for start, end, bit in segments:
            for step in range(start, min(end + 1, steps)):
                pair = "w%da%d" % (w, bit)
                slots.append(dict(layer_name=name, step_idx=step, weight_bit=w, act_bit=bit,
                                  pair=pair, n_params=count, mode="joint-wa", status="planned"))
                pairs[pair] += 1
                param_pairs[pair] += count
            if start < steps:
                seg_pairs["w%da%d" % (w, bit)] += 1
    slots.sort(key=lambda r: (r["layer_name"], r["step_idx"]))
    total, parameters, segments = len(slots), sum(param_pairs.values()), sum(seg_pairs.values())
    if not total:
        raise ValueError("No scheduled activation steps")
    ratio = []
    for pair in sorted(pairs, key=lambda p: (int(p[1:p.index('a')]), int(p[p.index('a') + 1:]))):
        w, a = int(pair[1:pair.index('a')]), int(pair[pair.index('a') + 1:])
        ratio.append(dict(pair=pair, weight_bit=w, act_bit=a, slot_count=pairs[pair],
                          slot_ratio=pairs[pair] / total, param_step_count=param_pairs[pair],
                          param_step_ratio=param_pairs[pair] / parameters,
                          segment_count=seg_pairs[pair], segment_ratio=seg_pairs[pair] / segments, status="planned"))
    result = dict(layers=len(schedule), slots=total, segments=segments,
                  nonuniform_layers=sum(len({b for _, _, b in s}) > 1 for s in schedule.values()),
                  slot_mean_A=sum(r["act_bit"] for r in slots) / total,
                  parameter_step_mean_A=sum(r["act_bit"] * r["n_params"] for r in slots) / parameters,
                  parameter_step_mean_W=sum(r["weight_bit"] * r["n_params"] for r in slots) / parameters,
                  pairs=ratio, scope="Planned per-image table counts; actual executed layer-step counts are recorded in inference.json")
    if output is not None:
        from pathlib import Path
        write_csv(Path(output) / "weight_activation_bits.csv", slots)
        write_csv(Path(output) / "weight_activation_ratios.csv", ratio)
        write_json(Path(output) / "activation_allocation.json", result)
    return result
