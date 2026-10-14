"""LPIPS and no-reference PyIQA evaluation for the four benchmarks."""
import math
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from .config import METRICS
from .io import images, write_csv, write_json


_OPTIONS = {"lpips": dict(net="alex", version="0.1")}


def _tensor(path, device):
    from torchvision.transforms.functional import to_tensor
    with Image.open(path) as image:
        return to_tensor(image.convert("RGB")).unsqueeze(0).to(device)


def summarize(rows, names):
    names = [name for name in names if name in METRICS]
    if not names:
        raise ValueError("No supported metric columns")
    per_image = [row for row in rows if row["image"] != "__DATASET_LEVEL__"]
    mean, counts = {}, {}
    for name in names:
        values = [float(row[name]) for row in per_image if row.get(name, "") != ""]
        if any(not math.isfinite(v) for v in values):
            raise ValueError("Nonfinite metric value: " + name)
        mean[name] = sum(values) / len(values) if values else None
        counts[name] = len(values)
    return dict(images=len(per_image), mean=mean, counts=counts)


def pair_images(paths, gt_dir, names, upscale=1):
    if "lpips" not in names:
        return {}
    if gt_dir is None:
        raise ValueError("Paired HR images are required by LPIPS")
    gt = {p.stem: p for p in images(gt_dir)}
    missing = [p.stem for p in paths if p.stem not in gt]
    if missing:
        raise ValueError("Missing paired HR images: " + ", ".join(missing))
    for path in paths:
        with Image.open(path) as sr, Image.open(gt[path.stem]) as hr:
            if (sr.width * upscale, sr.height * upscale) != hr.size:
                raise ValueError("SR/HR size mismatch for " + path.stem)
    return gt


@torch.inference_mode()
def evaluate(sr_dir, gt_dir, output, names, device="cuda:0", limit=None):
    if not names:
        raise ValueError("No metrics selected")
    paths = images(sr_dir, limit)
    unsupported = set(names) - METRICS
    if unsupported:
        raise ValueError("Unsupported metrics: " + ", ".join(sorted(unsupported)))
    gt = pair_images(paths, gt_dir, names)
    import pyiqa
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = [dict(image=str(p.relative_to(sr_dir)), **{name: "" for name in names}) for p in paths]
    metadata = dict(status="running", metrics=names,
                    options={name: _OPTIONS[name] for name in names if name in _OPTIONS},
                    device=str(device), pairing="unique filename stem" if gt else None)
    try:
        for name in names:
            metric = pyiqa.create_metric(name, device=device, **_OPTIONS.get(name, {}))
            for path, row in tqdm(list(zip(paths, rows)), desc=name, mininterval=2):
                sr = _tensor(path, device)
                score = metric(sr, _tensor(gt[path.stem], device)) if name == "lpips" else metric(sr)
                value = float(score.item())
                if not math.isfinite(value):
                    raise ValueError("Nonfinite %s for %s" % (name, path.name))
                row[name] = value
            del metric
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
        metadata["status"] = "complete"
    except Exception as exc:
        metadata.update(status="failed", error=str(exc))
        raise
    finally:
        write_csv(output / "metrics.csv", rows, ["image"] + names)
        summary = summarize(rows, names)
        summary.update(metadata)
        write_json(output / "summary.json", summary)
    return summary
