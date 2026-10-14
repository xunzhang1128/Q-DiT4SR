"""Quantize once, then super-resolve a sorted image sequence."""
import gc
import json
import time
from pathlib import Path

import torch
from accelerate.utils import set_seed
from PIL import Image

from .io import images, read_layer_table, sha256, write_csv, write_json
from .statistics import ActivationVarianceRecorder, load_transformer
from .quantization.linear import quantize
from .quantization.allocation import summarize_bits
from .quantization.activation import ActivationBitScheduler, load_schedule, summarize_schedule


def _condition(image, upscale, process_size, alignment=8):
    width, height = image.size
    minimum = max(1, process_size // upscale)
    if min(width, height) < minimum:
        scale = minimum / min(width, height)
        image = image.resize((max(8, round(width * scale)), max(8, round(height * scale))), Image.Resampling.BICUBIC)
    image = image.resize((image.width * upscale, image.height * upscale), Image.Resampling.BICUBIC)
    # Match the reference VAE input; square latent tiles handle odd grid extents.
    size = (max(alignment, image.width // alignment * alignment),
            max(alignment, image.height // alignment * alignment))
    return image.resize(size, Image.Resampling.BICUBIC)


def _rng_state():
    return dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def _restore_rng(state):
    if len(state["cuda"]) != torch.cuda.device_count():
        raise ValueError("RNG replay requires the same number of visible CUDA devices")
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state_all(state["cuda"])


def _prepare(config, cached, seed, replay=None):
    from .captions import Captioner
    from .pipeline import load_pipeline
    if torch.device(config["device"]).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("FP16 quantized inference requires CUDA")
    output = Path(config["output"])
    required = ["transformer", "base_model"] + (["caption_model", "clip_model"] if cached is None else [])
    for key in required:
        if not Path(config[key]).is_dir():
            raise FileNotFoundError("Missing local model directory: " + config[key])
    # The old module loads LLaVA before main() applies a seed.
    captioner = Captioner(config["caption_model"], config["clip_model"], config["caption_device"]) if cached is None else None
    if seed is not None:
        set_seed(seed)
    if replay is not None:
        _restore_rng(replay)
    setup_rng = _rng_state()
    bits = read_layer_table(config["weight_bits"], with_bits=True)
    if config["quantized_model"]:
        from .checkpoint import load_quantized
        transformer = load_quantized(config["quantized_model"], config["device"])
        saved = json.loads((Path(config["quantized_model"]) / "quantization.json").read_text())
        layers = saved["layers"]
        statistics = [dict(layer_name=r["layer_name"], variance=r["variance"],
                           n_params=r["in_features"] * r["out_features"]) for r in layers]
    else:
        transformer = load_transformer(config["transformer"])
        statistics, layers = quantize(transformer, bits, config, config["device"])
    write_csv(output / "weight_statistics.csv", statistics)
    write_csv(output / "weight_bits.csv", bits, ["layer_name", "variance", "n_params", "bits_real", "bits_assigned"])
    write_json(output / "quantization.json", dict(runtime="FP16 simulated quantization", quantizer="RobustUQ", hadamard=True,
               bit_allocation=summarize_bits(bits), transformer_parameters=sum(p.numel() for p in transformer.parameters()), layers=layers))
    pipeline = load_pipeline(config["base_model"], transformer).to(config["device"])
    if pipeline.scheduler.__class__.__name__ != "FlowMatchEulerDiscreteScheduler":
        raise ValueError("The default denoising loop requires FlowMatchEulerDiscreteScheduler")
    return pipeline, captioner, bits, setup_rng


def _captions(config, paths):
    cached = json.loads(Path(config["captions"]).read_text()) if config["captions"] else None
    if cached is not None:
        missing = [p.name for p in paths if p.name not in cached]
        if missing or any(not isinstance(cached[p.name], str) or not cached[p.name].strip() for p in paths if p.name in cached):
            raise ValueError("Cached captions must map every input filename to a nonempty raw caption: " + str(missing))
    return cached


def infer(config, limit=None):
    from .captions import filter_caption
    from .pipeline import adain_color_fix
    paths = images(config["input"], limit)
    output, sr_root = Path(config["output"]), Path(config["output"]) / "sr"
    if sr_root.exists() and any(sr_root.iterdir()):
        raise FileExistsError("Inference output already contains images; choose a new --output: " + str(sr_root))
    cached = _captions(config, paths)
    replay = torch.load(config["rng_state"], map_location="cpu", weights_only=True) if config["rng_state"] else None
    expected = replay["sequence"][:len(paths)] if replay is not None and limit is not None else (
        replay["sequence"] if replay is not None else None)
    if replay is not None and (cached is None or expected != [p.name for p in paths]):
        raise ValueError("RNG replay requires saved captions and the original sorted input sequence")
    pipeline, captioner, bits, setup_rng = _prepare(config, cached, config["seed"], replay["setup"] if replay else None)
    schedule = load_schedule(config["activation_schedule"], config["steps"])
    if schedule:
        summarize_schedule(schedule, bits, config["steps"], output)
    scheduler = ActivationBitScheduler(schedule, config["a_bits"]) if schedule else None
    schedule_info = scheduler.attach(pipeline.transformer) if scheduler else dict(mode="uniform", a_bits=config["a_bits"])
    write_json(output / "activation_scheduler.json", schedule_info)
    generator = torch.Generator(device=config["device"])
    if config["seed"] is not None:
        generator.manual_seed(config["seed"])
    rng = dict(setup=setup_rng, sequence=[p.name for p in paths], images={})
    captions, prompts, records = {}, {}, []
    sr_root.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(paths):
        print("[%d/%d] %s" % (index + 1, len(paths), path.name), flush=True)
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        caption = cached[path.name] if cached is not None else captioner(image)
        prompt = filter_caption(caption) + " " + config["positive_suffix"]
        captions[path.name], prompts[path.name] = caption, prompt
        write_json(output / "captions.json", captions)
        write_json(output / "prompts.json", prompts)
        condition = _condition(image, config["upscale"], config["process_size"], config["image_alignment"])
        if replay is not None:
            state = replay["images"][path.name]
            if state["input_sha256"] != sha256(path):
                raise ValueError("RNG replay input changed: " + path.name)
            _restore_rng(state)
            generator.set_state(state["generator"])
        rng["images"][path.name] = dict(**_rng_state(), generator=generator.get_state(), input_sha256=sha256(path))
        torch.save(rng, output / "rng_states.pt")
        if scheduler:
            scheduler.start_image()
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            result = pipeline(condition, prompt, config["negative_prompt"], generator,
                              steps=config["steps"], guidance_scale=config["guidance_scale"],
                              tile_size=config["tile_size"], tile_overlap=config["tile_overlap"],
                              vae_input_layout=config["vae_input_layout"])
        result = adain_color_fix(result, condition)
        result = result.resize((image.width * config["upscale"], image.height * config["upscale"]), Image.Resampling.BICUBIC)
        destination = sr_root / (path.stem + ".png")
        result.save(destination)
        records.append(dict(input=str(path), input_sha256=sha256(path), output=str(destination),
                            output_sha256=sha256(destination), control_size=list(condition.size),
                            output_size=list(result.size), seconds=time.perf_counter() - start,
                            activation_scheduler_step=scheduler.step_idx if scheduler else None,
                            activation_bit_counts=dict(scheduler.bit_counts) if scheduler else None))
        write_json(output / "inference.json", dict(status="running", images=records))
    write_json(output / "inference.json", dict(status="complete", images=records))
    if scheduler:
        scheduler.detach()
    del pipeline, captioner
    gc.collect()
    torch.cuda.empty_cache()
    return sr_root


def calibration_images(config):
    generator = torch.Generator().manual_seed(config["calibration_seed"])
    paths = []
    directories = [config["calibration_canon"]]
    if config["calibration_nikon"] is not None:
        directories.append(config["calibration_nikon"])
    count = config["calibration_samples"] // len(directories)
    for directory in directories:
        pool = images(directory)
        if len(pool) < count:
            raise ValueError("Not enough calibration images: expected %d in %s" % (count, directory))
        selected = torch.randperm(len(pool), generator=generator)[:count].tolist() if count < len(pool) else range(len(pool))
        paths.extend(pool[i] for i in selected)
    return paths


def collect_activation_statistics(config):
    from .captions import filter_caption
    paths = calibration_images(config)
    cached = _captions(config, paths)
    output = Path(config["output"])
    pipeline, captioner, _, _ = _prepare(config, cached, config["calibration_seed"])
    pipeline.set_progress_bar_config(disable=True)
    recorder = ActivationVarianceRecorder()
    recorder.attach(pipeline.transformer)
    generator = torch.Generator(device=config["device"]).manual_seed(config["calibration_seed"])
    captions = {}
    write_json(output / "calibration_images.json", [dict(path=str(p), sha256=sha256(p)) for p in paths])
    try:
        for index, path in enumerate(paths):
            print("[Calibration %d/%d] %s" % (index + 1, len(paths), path.name), flush=True)
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            caption = cached[path.name] if cached is not None else captioner(image)
            captions[path.name] = caption
            write_json(output / "captions.json", captions)
            prompt = filter_caption(caption) + " " + config["positive_suffix"]
            condition = _condition(image, config["upscale"], config["process_size"], config["image_alignment"])
            recorder.start_image()
            with torch.autocast("cuda", dtype=torch.float16):
                pipeline(condition, prompt, config["negative_prompt"], generator,
                         steps=config["steps"], guidance_scale=config["guidance_scale"],
                         tile_size=config["tile_size"], tile_overlap=config["tile_overlap"],
                         vae_input_layout=config["vae_input_layout"])
            write_csv(output / "activation_statistics.csv", recorder.rows())
    finally:
        recorder.detach()
        del pipeline, captioner
        gc.collect()
        torch.cuda.empty_cache()
    result = dict(images=len(paths), layers=len({r["layer_name"] for r in recorder.rows()}), rows=len(recorder.rows()),
                  steps=config["steps"], seed=config["calibration_seed"], mode="post-Hadamard pre-quantization token variance",
                  normalization_observers_frozen=False)
    write_json(output / "activation_statistics.json", result)
    return result
