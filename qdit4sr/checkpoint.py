"""Save frozen quantized weights; restore their branches without decomposition."""
import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file, save_file
from torch import nn

from .io import read_layer_table, sha256, write_json
from .models import SD3Transformer2DModel
from .quantization.blr import BlockLowRank, block_config
from .quantization.linear import GlobalLowRank, MixedLinear, quantize, target_linears
from .quantization.uniform import RobustUQ


def _empty_layer(linear, spec):
    """Allocate only shapes on meta; no SVD, BLR search or random transforms."""
    q = MixedLinear.__new__(MixedLinear)
    nn.Linear.__init__(q, linear.in_features, linear.out_features,
                       bias=linear.bias is not None, device="meta", dtype=torch.float16)
    if (q.in_features, q.out_features) != (spec["in_features"], spec["out_features"]):
        raise ValueError("Quantized layer shape mismatch: " + spec["layer_name"])
    q.w_bits, q.a_bits, q.variance = spec["w_bits"], spec["a_bits"], spec["variance"]
    q.register_buffer("signs", torch.empty(q.in_features, device="meta", dtype=torch.float16))
    q.svd = None
    if spec["svd_rank"]:
        q.svd = GlobalLowRank.__new__(GlobalLowRank)
        nn.Module.__init__(q.svd)
        q.svd.rank = spec["svd_rank"]
        q.svd.B = nn.Linear(q.in_features, q.svd.rank, bias=False, device="meta", dtype=torch.float16)
        q.svd.A = nn.Linear(q.svd.rank, q.out_features, bias=False, device="meta", dtype=torch.float16)
    q.blr = None
    if spec["blr_parameters"]:
        q.blr = BlockLowRank.__new__(BlockLowRank)
        nn.Module.__init__(q.blr)
        q.blr.register_parameter("bias", None)
        q.blr.in_features, q.blr.out_features = q.in_features, q.out_features
        q.blr.bo, q.blr.bi = spec["blr_block"]
        q.blr.mo, q.blr.mi = spec["blr_grid"]
        q.blr.rank = spec["blr_block_rank"]
        q.blr.U = nn.Parameter(torch.empty(q.blr.mo, q.blr.mi, q.blr.bo, q.blr.rank,
                                          device="meta", dtype=torch.float16))
        q.blr.V = nn.Parameter(torch.empty(q.blr.mo, q.blr.mi, q.blr.rank, q.blr.bi,
                                          device="meta", dtype=torch.float16))
    q.weight_quantizer = nn.Identity()
    q.activation_quantizer = RobustUQ(q.a_bits).half().to("meta")
    q.post_hadamard_tap = None
    return q


@torch.no_grad()
def save_quantized(model, output, config, layers):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "model.safetensors").exists():
        raise FileExistsError("Quantized checkpoint already exists: " + str(output))
    by_name = {entry["layer_name"]: dict(entry) for entry in layers}
    targets = {n: q for n, q in model.named_modules() if isinstance(q, MixedLinear)}
    if set(targets) != set(by_name):
        raise ValueError("Expected a complete quantized Transformer")
    # Weight quantization is input-independent. Keep the exact FP16 STE result.
    for name, q in targets.items():
        q.weight.copy_(q.weight_quantizer(q.weight))
        q.weight_quantizer = nn.Identity()
        by_name[name]["variance"] = q.variance
    model.save_config(output)
    architecture = json.loads((output / "config.json").read_text())
    architecture.pop("_name_or_path", None)
    write_json(output / "config.json", architecture)
    temporary = output / "model.safetensors.tmp"
    save_file({n: t.detach().cpu().contiguous() for n, t in model.state_dict().items()},
              str(temporary), metadata={"format": "pt"})
    temporary.replace(output / "model.safetensors")
    portable = {key: config[key] for key in ("setting", "a_bits", "svd_rank", "blr_rank_budget", "seed", "steps",
                "guidance_scale", "upscale", "process_size", "image_alignment", "vae_input_layout",
                "tile_size", "tile_overlap", "positive_suffix", "negative_prompt")}
    portable.update(quantized_model=".", activation_schedule_required=bool(config["activation_schedule"]))
    write_json(output / "pipeline_config.json", portable)
    metadata = dict(format_version=1, storage="FP16 frozen RobustUQ weights and FP16 SVD/BLR branches",
                    parameters=sum(p.numel() for p in model.parameters()), layers=list(by_name.values()),
                    model_sha256=sha256(output / "model.safetensors"))
    write_json(output / "quantization.json", metadata)
    return metadata


def load_quantized(path, device="cpu"):
    path = Path(path)
    metadata = json.loads((path / "quantization.json").read_text())
    if metadata["format_version"] != 1:
        raise ValueError("Unsupported quantized checkpoint format")
    if sha256(path / "model.safetensors") != metadata["model_sha256"]:
        raise ValueError("Quantized checkpoint SHA256 does not match its metadata")
    specs = {entry["layer_name"]: entry for entry in metadata["layers"]}
    pipeline = json.loads((path / "pipeline_config.json").read_text())
    if any(type(spec["w_bits"]) is not int or not 1 <= spec["w_bits"] <= 8 or
           spec["a_bits"] != pipeline["a_bits"] for spec in specs.values()):
        raise ValueError("Frozen weight/activation bits do not match the model bundle")
    if (path / "weight_bits.csv").exists():
        rows = {r["layer_name"]: r for r in read_layer_table(path / "weight_bits.csv", with_bits=True)}
        if set(rows) != set(specs) or any(
            rows[name]["bits_assigned"] != spec["w_bits"] or
            rows[name]["n_params"] != spec["in_features"] * spec["out_features"]
            for name, spec in specs.items()
        ):
            raise ValueError("Frozen weight/activation bits do not match the model bundle")
    for spec in specs.values():
        expected_rank = min(pipeline["svd_rank"], spec["in_features"], spec["out_features"])
        if spec["svd_rank"] != expected_rank:
            raise ValueError("Frozen SVD rank does not match the model bundle")
        if pipeline["blr_rank_budget"]:
            expected_block = block_config(spec["in_features"], spec["out_features"], pipeline["blr_rank_budget"])
            if tuple(spec.get("blr_block", [])) + (spec.get("blr_block_rank"),) != expected_block:
                raise ValueError("Frozen BLR configuration does not match the model bundle")
        elif spec["blr_parameters"]:
            raise ValueError("Frozen BLR configuration does not match the model bundle")
    architecture = json.loads((path / "config.json").read_text())
    with init_empty_weights(include_buffers=True):
        model = SD3Transformer2DModel.from_config(architecture)
    targets = target_linears(model)
    if len(specs) != len(metadata["layers"]) or set(specs) != set(targets):
        raise ValueError("Quantized checkpoint does not cover the Transformer layers")
    for name, linear in targets.items():
        parent, leaf = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), leaf, _empty_layer(linear, specs[name]))
    weights = load_file(str(path / "model.safetensors"), device="cpu")
    model.load_state_dict(weights, strict=True, assign=True)
    return model.to(device).eval().requires_grad_(False)


def export(config):
    from accelerate.utils import set_seed
    from .runtime import _restore_rng
    from .statistics import load_transformer
    if torch.device(config["device"]).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Export the validated FP16 quantization on CUDA")
    if config["quantized_model"]:
        raise ValueError("Export requires the original trained Transformer")
    output = Path(config["output"])
    if (output / "model.safetensors").exists():
        raise FileExistsError("Quantized checkpoint already exists: " + str(output))
    if config["seed"] is not None:
        set_seed(config["seed"])
    if config["rng_state"]:
        _restore_rng(torch.load(config["rng_state"], map_location="cpu", weights_only=True)["setup"])
    model = load_transformer(config["transformer"])
    bits = read_layer_table(config["weight_bits"], with_bits=True)
    _, layers = quantize(model, bits, config, config["device"])
    result = save_quantized(model, output, config, layers)
    result["source_checkpoint_sha256"] = sha256(Path(config["transformer"]) / "diffusion_pytorch_model.safetensors")
    write_json(output / "quantization.json", result)
    return {k: v for k, v in result.items() if k != "layers"}
