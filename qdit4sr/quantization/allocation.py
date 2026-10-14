"""Parameter-weighted variance allocation; the original default greedy solver."""
import heapq

import numpy as np


def excluded(row):
    name = row["layer_name"]
    return row["variance"] <= 1e-9 or ("attn2" in name and any(
        part in name for part in ("add_q_proj", "add_k_proj", "add_v_proj", "to_add_out")))


def allocate(rows, target=4.0, min_bits=2, max_bits=8):
    if not 1 <= min_bits <= target <= max_bits <= 8:
        raise ValueError("Expected 1 <= min_bits <= target <= max_bits <= 8")
    result = [dict(row, bits_real=1.0, bits_assigned=1) for row in rows]
    indices = [i for i, row in enumerate(rows) if not excluded(row)]
    if not indices:
        return result
    variance = np.array([rows[i]["variance"] for i in indices], dtype=float)
    parameters = np.array([rows[i]["n_params"] for i in indices], dtype=float)
    log = np.log2(np.maximum(variance, 1e-12))
    real = target + 0.5 * (log - np.average(log, weights=parameters))
    bits = np.clip(np.floor(real).astype(np.int32), min_bits, max_bits)
    budget, current = target * parameters.sum(), (bits * parameters).sum()
    queue = [(-variance[i] * 4.0 ** (-int(bits[i])), i)
             for i in range(len(indices)) if bits[i] < max_bits]
    heapq.heapify(queue)
    while current < budget:
        if not queue:
            raise ValueError("Cannot reach the requested active-layer budget")
        _, index = heapq.heappop(queue)
        bits[index] += 1
        current += parameters[index]
        if bits[index] < max_bits:
            heapq.heappush(queue, (-variance[index] * 4.0 ** (-int(bits[index])), index))
    for position, index in enumerate(indices):
        result[index].update(bits_real=float(real[position]), bits_assigned=int(bits[position]))
    return result


def summarize_bits(rows):
    active = [row for row in rows if not excluded(row)]
    histogram = {}
    for row in rows:
        item = histogram.setdefault(str(row["bits_assigned"]), dict(layers=0, parameters=0))
        item["layers"] += 1
        item["parameters"] += row["n_params"]
    def average(subset):
        total = sum(r["n_params"] for r in subset)
        return sum(r["bits_assigned"] * r["n_params"] for r in subset) / total if total else None
    return dict(layers=len(rows), active_layers=len(active), excluded_layers=len(rows) - len(active),
                dense_parameters=sum(r["n_params"] for r in rows),
                active_weighted_bits=average(active), overall_weighted_bits=average(rows), histogram=histogram)
