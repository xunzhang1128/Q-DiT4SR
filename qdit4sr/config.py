import json
import math
from pathlib import Path


_PATHS = {"transformer", "base_model", "caption_model", "clip_model", "input", "gt", "output", "captions", "weight_bits",
          "activation_schedule", "activation_statistics", "calibration_canon", "calibration_nikon", "rng_state", "quantized_model"}
_WORKSPACE = "../Q-DiT4SR_workspace"
_NR_METRICS = ["musiq", "maniqa", "clipiqa+_vitL14_512", "liqe"]
BENCHMARKS = {
    "RealSR": dict(input=_WORKSPACE + "/data/RealSR/test_LR", gt=_WORKSPACE + "/data/RealSR/test_HR",
                   metrics=["lpips"] + _NR_METRICS),
    "DrealSR": dict(input=_WORKSPACE + "/data/DrealSR/test_LR", gt=_WORKSPACE + "/data/DrealSR/test_HR",
                    metrics=["lpips"] + _NR_METRICS),
    "RealLQ250": dict(input=_WORKSPACE + "/data/RealLQ250/lq", gt=None, metrics=list(_NR_METRICS)),
    "RealLR200": dict(input=_WORKSPACE + "/data/RealLR200", gt=None, metrics=list(_NR_METRICS)),
}
METRICS = {name for benchmark in BENCHMARKS.values() for name in benchmark["metrics"]}


def load_config(path=None, overrides=None):
    overrides = {key: value for key, value in (overrides or {}).items() if value is not None}
    supplied = json.loads(Path(path).read_text()) if path else {}
    config = dict(
        setting="w4a4", benchmark="RealSR", quantized_model=None,
        transformer=_WORKSPACE + "/models/dit4sr/transformer",
        base_model=_WORKSPACE + "/models/stable-diffusion-3.5-medium",
        caption_model=_WORKSPACE + "/models/llava-v1.5-13b",
        clip_model=_WORKSPACE + "/models/clip-vit-large-patch14-336",
        input=BENCHMARKS["RealSR"]["input"], gt=BENCHMARKS["RealSR"]["gt"], output="outputs/default",
        weight_bits=None, captions=None, device="cuda:0", caption_device="cuda:0",
        a_bits=4, svd_rank=32, blr_rank_budget=16, seed=None, steps=40, guidance_scale=8.0,
        upscale=4, process_size=512, image_alignment=8, vae_input_layout="contiguous",
        tile_size=64, tile_overlap=24,
        positive_suffix="Cinematic, hyper sharpness, highly detailed, perfect without deformations, camera, "
                        "hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, "
                        "extreme meticulous detailing, skin pore detailing. ",
        negative_prompt="motion blur, noisy, dotted, bokeh, pointed, CG Style, 3D render, unreal engine, "
                        "blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, "
                        "jpeg artifacts, deformed, lowres, chaotic",
        metrics=list(BENCHMARKS["RealSR"]["metrics"]), calibration_seed=0, calibration_samples=32,
        calibration_canon="data/calibration", calibration_nikon=None,
        activation_schedule=None, activation_schedule_required=False, activation_statistics=None, rng_state=None,
    )
    setting = overrides.get("setting", supplied.get("setting", config["setting"]))
    if setting not in {"w4a4", "w4a6"}:
        raise ValueError("Unknown setting: " + str(setting))
    if setting == "w4a6":
        config.update(setting="w4a6", a_bits=6, blr_rank_budget=8, seed=42,
                      image_alignment=16, vae_input_layout="channels_last")
    root = Path.cwd()
    if path:
        path = Path(path).resolve()
        unknown = set(supplied) - set(config)
        if unknown:
            raise ValueError("Unknown config keys: " + ", ".join(sorted(unknown)))
        for key, value in supplied.items():
            if key in _PATHS and value is not None:
                value = str((path.parent / value).resolve())
            config[key] = value
    benchmark = overrides.get("benchmark", config["benchmark"])
    if benchmark not in BENCHMARKS:
        raise ValueError("Unknown benchmark: " + str(benchmark))
    if "benchmark" in overrides or "benchmark" in supplied:
        config.update(BENCHMARKS[benchmark])
        # Explicit paths in a dataset-specific config remain usable.
        if "benchmark" not in overrides:
            for key in ("input", "gt", "metrics"):
                if key in supplied:
                    config[key] = str((path.parent / supplied[key]).resolve()) if key in _PATHS and supplied[key] is not None else supplied[key]
    config.update(overrides)
    for key in _PATHS:
        if config[key] is not None:
            config[key] = str((root / config[key]).resolve())
    if config["quantized_model"]:
        bundle = Path(config["quantized_model"])
        saved = json.loads((bundle / "pipeline_config.json").read_text())
        requested_setting = overrides.get("setting", supplied.get("setting"))
        if requested_setting is not None and saved["setting"] != requested_setting:
            raise ValueError("Selected setting does not match the saved quantized model")
        for key in ("setting", "a_bits", "svd_rank", "blr_rank_budget", "steps", "guidance_scale", "upscale",
                    "process_size", "tile_size", "tile_overlap", "positive_suffix", "negative_prompt"):
            config[key] = saved[key]
        for key, legacy_default in (("image_alignment", 8), ("vae_input_layout", "contiguous")):
            config[key] = saved.get(key, legacy_default)
        if "seed" not in supplied and "seed" not in overrides:
            config["seed"] = saved["seed"]
        config["transformer"] = str(bundle)
        config["weight_bits"] = str(bundle / "quantization.json")
        if "activation_schedule" not in supplied and "activation_schedule" not in overrides:
            schedule = bundle / saved["activation_schedule"] if saved.get("activation_schedule") else None
            config["activation_schedule"] = str(schedule) if schedule is not None and schedule.is_file() else None
        config["activation_schedule_required"] = saved.get("activation_schedule_required", bool(saved.get("activation_schedule")))
        config["activation_statistics"] = None
    for key in ("steps", "upscale", "process_size", "tile_size"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(key + " must be a positive integer")
    if config["process_size"] % 16 or config["tile_size"] % 2:
        raise ValueError("process_size must be divisible by 16; tile_size must be even")
    if type(config["image_alignment"]) is not int or config["image_alignment"] not in {8, 16}:
        raise ValueError("image_alignment must be 8 or 16")
    if config["vae_input_layout"] not in ("contiguous", "channels_last"):
        raise ValueError("vae_input_layout must be contiguous or channels_last")
    for key in ("svd_rank", "blr_rank_budget"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(key + " must be a nonnegative integer")
    if type(config["a_bits"]) is not int or not 1 <= config["a_bits"] <= 8:
        raise ValueError("a_bits must be an integer in [1, 8]")
    if config["seed"] is not None and (type(config["seed"]) is not int or not 0 <= config["seed"] < 2 ** 32):
        raise ValueError("seed must be null or an integer in [0, 2^32)")
    if type(config["calibration_seed"]) is not int or not 0 <= config["calibration_seed"] < 2 ** 32:
        raise ValueError("calibration_seed must be an integer in [0, 2^32)")
    if type(config["calibration_samples"]) is not int or config["calibration_samples"] <= 0 or config["calibration_samples"] % 2:
        raise ValueError("calibration_samples must be a positive even integer")
    if not isinstance(config["guidance_scale"], (int, float)) or not math.isfinite(config["guidance_scale"]) or config["guidance_scale"] < 1:
        raise ValueError("guidance_scale must be finite and at least 1")
    if not 0 <= config["tile_overlap"] < config["tile_size"]:
        raise ValueError("tile_overlap must be smaller than tile_size")
    if not isinstance(config["metrics"], list) or not config["metrics"] or not all(isinstance(n, str) for n in config["metrics"]) or len(set(config["metrics"])) != len(config["metrics"]):
        raise ValueError("metrics must be a nonempty list without duplicates")
    unsupported = set(config["metrics"]) - set(BENCHMARKS[benchmark]["metrics"])
    if unsupported:
        raise ValueError("Unsupported metrics for %s: %s" % (benchmark, ", ".join(sorted(unsupported))))
    return config
