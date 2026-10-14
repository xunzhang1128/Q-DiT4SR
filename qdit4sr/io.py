import csv
import gzip
import hashlib
import json
from pathlib import Path


_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def open_text(path):
    return gzip.open(path, "rt", newline="", encoding="utf-8") if str(path).endswith(".gz") else Path(path).open(newline="", encoding="utf-8")


def images(root, limit=None):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError("Image directory does not exist: " + str(root))
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _EXTENSIONS)
    if not paths:
        raise ValueError("No images in " + str(root))
    stems = [p.stem for p in paths]
    if len(stems) != len(set(stems)):
        raise ValueError("Duplicate filename stems in " + str(root))
    if limit is not None:
        if limit <= 0:
            raise ValueError("Image limit must be positive")
        paths = paths[:limit]
    return paths


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_layer_table(path, with_bits=False):
    """Read generated CSV statistics or the bits stored in a frozen checkpoint."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        metadata = json.loads(path.read_text(encoding="utf-8"))
        rows = [dict(layer_name=layer["layer_name"], variance=layer["variance"],
                     n_params=layer["in_features"] * layer["out_features"],
                     bits_real=float(layer["w_bits"]), bits_assigned=layer["w_bits"])
                for layer in metadata["layers"]]
    else:
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Empty layer table: " + str(path))
    import math
    seen = set()
    for row in rows:
        name = row["layer_name"]
        if not name or name in seen:
            raise ValueError("Empty or duplicate layer name: " + name)
        seen.add(name)
        row["variance"] = float(row["variance"])
        row["n_params"] = int(row["n_params"])
        if not math.isfinite(row["variance"]) or row["variance"] < 0 or row["n_params"] <= 0:
            raise ValueError("Invalid statistics for " + name)
        if with_bits:
            row["bits_assigned"] = int(row["bits_assigned"])
            if not 1 <= row["bits_assigned"] <= 8:
                raise ValueError("Invalid weight bits for " + name)
            row["bits_real"] = float(row["bits_real"])
    return rows
