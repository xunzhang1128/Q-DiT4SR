import argparse
import csv
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import BENCHMARKS, load_config
from .io import images, read_layer_table, sha256, write_csv, write_json


def _parser():
    parser = argparse.ArgumentParser(description="Q-DiT4SR: mixed-precision quantization, SR and evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("stats", "Measure signed-Hadamard weight variance before SVD"),
        ("allocate", "Allocate mixed weight bits from variance"),
        ("act-stats", "Collect calibration activation variance with uniform A4"),
        ("schedule", "Allocate activation bits over diffusion steps"),
        ("profile", "Count Transformer parameters and executed Linear work"),
        ("export", "Save a quantized Transformer and portable configuration"),
        ("infer", "Load a frozen model or quantize the trained Transformer, then super-resolve"),
        ("evaluate", "Evaluate existing SR images"),
        ("summarize", "Aggregate per-image metrics from a CSV file"),
        ("run", "Quantize, collect statistics, infer and evaluate"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", help="JSON overrides; paths are relative to this file")
        command.add_argument("--output", help="Output directory")
        if name in {"stats", "act-stats", "profile", "infer", "run", "export"}:
            command.add_argument("--transformer", help="DiT4SR Transformer directory")
        if name in {"stats", "act-stats", "infer", "run", "evaluate", "export"}:
            command.add_argument("--device")
        if name in {"stats", "act-stats", "schedule", "profile", "infer", "run", "export"}:
            command.add_argument("--weight-bits", help="Complete layer bit CSV or quantization.json from a frozen model")
        if name in {"profile", "infer", "run", "export"}:
            command.add_argument("--setting", choices=("w4a4", "w4a6"))
            command.add_argument("--activation-schedule", help="Per-layer timestep activation CSV, optionally gzip")
        if name in {"profile", "infer", "run"}:
            command.add_argument("--quantized-model", help="Exported quantized Transformer directory; skips decomposition")
        if name in {"infer", "run", "evaluate"}:
            command.add_argument("--benchmark", choices=tuple(BENCHMARKS), help="Dataset paths and paired/no-reference metrics")
        if name in {"act-stats", "infer", "run"}:
            command.add_argument("--base-model")
            command.add_argument("--caption-model")
            command.add_argument("--clip-model")
            command.add_argument("--caption-device")
            command.add_argument("--captions", help="JSON mapping input filenames to raw captions")
        if name in {"infer", "run"}:
            command.add_argument("--input", help="LR directory")
        if name in {"infer", "run", "export"}:
            command.add_argument("--seed", type=int, help="Optional inference seed; default leaves global RNG unseeded")
            command.add_argument("--rng-state", help="Saved RNG states: export uses setup; inference replays images with --captions")
        if name == "act-stats":
            command.add_argument("--calibration-canon")
            command.add_argument("--calibration-nikon")
            command.add_argument("--calibration-seed", type=int, help="Calibration sampling and quantization seed")
        if name == "run":
            command.add_argument("--calibrate", action="store_true", help="Regenerate statistics and schedule in a separate calibration process")
        if name in {"run", "evaluate"}:
            command.add_argument("--gt", help="Paired HR directory")
            command.add_argument("--metrics", help="Comma-separated subset of the benchmark's metrics")
        if name in {"infer", "run", "evaluate"}:
            command.add_argument("--limit", type=int, help="Process the first N sorted images")
        if name == "evaluate":
            command.add_argument("--sr", required=True, help="SR directory")
        if name == "allocate":
            command.add_argument("--statistics", required=True, help="Weight variance CSV from stats")
            command.add_argument("--target", type=float, default=4.0)
            command.add_argument("--min-bits", type=int, default=2)
            command.add_argument("--max-bits", type=int, default=8)
        if name == "schedule":
            command.add_argument("--statistics", required=True, help="Activation variance CSV from act-stats")
        if name == "summarize":
            command.add_argument("--csv", required=True)
    return parser


def _versions():
    result = {"python": sys.version.split()[0]}
    for name in ("qdit4sr", "torch", "torchvision", "diffusers", "transformers", "accelerate", "pyiqa", "Pillow",
                 "bitsandbytes", "numpy", "pandas", "scipy", "timm", "safetensors", "fast-hadamard-transform"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result


def main(argv=None):
    args = _parser().parse_args(argv)
    keys = ("transformer", "base_model", "caption_model", "clip_model", "input", "gt", "output",
            "device", "caption_device", "captions", "weight_bits", "activation_schedule", "rng_state", "seed",
            "calibration_canon", "calibration_nikon", "calibration_seed", "setting", "benchmark", "quantized_model")
    overrides = {key: getattr(args, key, None) for key in keys}
    if getattr(args, "metrics", None):
        overrides["metrics"] = [name.strip() for name in args.metrics.split(",") if name.strip()]
    try:
        config = load_config(args.config, overrides)
        if args.command in {"act-stats", "schedule", "profile", "infer", "run", "export"} and not config["weight_bits"]:
            raise ValueError("Provide --weight-bits for the original Transformer, or load --quantized-model")
        if args.command in {"profile", "infer", "run"} and config["activation_schedule_required"] and not config["activation_schedule"]:
            raise ValueError("This checkpoint requires --activation-schedule; provide the bundled CSV or generate one with act-stats and schedule")
        output = Path(config["output"])
        output.mkdir(parents=True, exist_ok=True)
        record = dict(command=args.command, arguments=vars(args), config=config,
                      started=datetime.now(timezone.utc).isoformat(), status="running", versions=_versions())
        record["code_sha256"] = {str(p.relative_to(Path(__file__).parent)): sha256(p)
                                 for p in sorted(Path(__file__).parent.rglob("*.py"))}
        record["weight_bits_sha256"] = sha256(config["weight_bits"]) if config["weight_bits"] else None
        record["activation_schedule_sha256"] = sha256(config["activation_schedule"]) if config["activation_schedule"] else None
        manifest = output / (args.command + "_manifest.json")
        # Keep an earlier completed run's provenance intact when rerunning a command.
        if manifest.exists():
            raise FileExistsError("Manifest already exists; choose a new --output: " + str(manifest))
        write_json(manifest, record)
        try:
            if args.command == "run":
                from .evaluation import pair_images
                pair_images(images(config["input"], args.limit), config["gt"], config["metrics"], config["upscale"])
            if args.command == "stats":
                from .statistics import weight_statistics
                layer_order = ([r["layer_name"] for r in read_layer_table(config["weight_bits"], with_bits=True)]
                               if config["weight_bits"] else None)
                rows, details = weight_statistics(config["transformer"], config["device"],
                                                 config["calibration_seed"], layer_order)
                write_csv(output / "weight_statistics.csv", rows)
                write_json(output / "statistics.json", details)
                result = dict(layers=len(rows), **details)
            elif args.command == "allocate":
                from .quantization.allocation import allocate, summarize_bits
                rows = allocate(read_layer_table(args.statistics), args.target, args.min_bits, args.max_bits)
                write_csv(output / "weight_bits.csv", rows)
                result = summarize_bits(rows)
                write_json(output / "allocation.json", result)
                record["statistics_sha256"] = sha256(args.statistics)
            elif args.command == "act-stats":
                from .runtime import collect_activation_statistics
                record["checkpoint_sha256"] = sha256(Path(config["transformer"]) / "diffusion_pytorch_model.safetensors")
                result = collect_activation_statistics(config)
            elif args.command == "schedule":
                from .quantization.activation import allocate_activation, load_schedule, summarize_schedule
                source = args.statistics
                result = allocate_activation(source, output, target=config["a_bits"])
                summarize_schedule(load_schedule(output / "activation_schedule.csv"),
                                   read_layer_table(config["weight_bits"], with_bits=True), config["steps"], output)
                record["statistics_sha256"] = sha256(source)
            elif args.command == "export":
                from .checkpoint import export
                result = export(config)
            elif args.command in {"profile", "infer", "run"}:
                if args.command == "run" and args.calibrate:
                    if config["quantized_model"]:
                        raise ValueError("Regenerate calibration from the original Transformer before exporting")
                    calibration = output / "calibration"
                    write_json(calibration / "config.json", config)
                    subprocess.run([sys.executable, "-m", "qdit4sr", "act-stats", "--config",
                                    str(calibration / "config.json"), "--output", str(calibration)], check=True)
                    from .quantization.activation import allocate_activation
                    config["activation_statistics"] = str(calibration / "activation_statistics.csv")
                    allocate_activation(config["activation_statistics"], calibration, config["a_bits"])
                    config["activation_schedule"] = str(calibration / "activation_schedule.csv")
                    record["statistics_sha256"] = sha256(config["activation_statistics"])
                    record["activation_schedule_sha256"] = sha256(config["activation_schedule"])
                from .statistics import profile
                result = profile(config["transformer"], config["weight_bits"], config, steps=config["steps"])
                write_json(output / "profile.json", result)
                checkpoint_name = "model.safetensors" if config["quantized_model"] else "diffusion_pytorch_model.safetensors"
                record["checkpoint_sha256"] = sha256(Path(config["transformer"]) / checkpoint_name)
                if args.command in {"infer", "run"}:
                    model_configs = [Path(config["transformer"]) / "config.json",
                                     Path(config["base_model"]) / "scheduler" / "scheduler_config.json"]
                    model_configs += [Path(config["base_model"]) / part / "config.json"
                                      for part in ("vae", "text_encoder", "text_encoder_2", "text_encoder_3")]
                    if config["captions"]:
                        record["captions_sha256"] = sha256(config["captions"])
                    else:
                        model_configs += [Path(config[key]) / "config.json" for key in ("caption_model", "clip_model")]
                    if config["rng_state"]:
                        record["rng_state_sha256"] = sha256(config["rng_state"])
                    record["model_config_sha256"] = {str(path): sha256(path) for path in model_configs}
                    from .runtime import infer
                    sr = infer(config, args.limit)
                    result = dict(sr=str(sr))
                    if args.command == "run":
                        from .evaluation import evaluate
                        result = evaluate(str(sr), config["gt"], output, config["metrics"], config["device"])
            elif args.command == "evaluate":
                from .evaluation import evaluate
                result = evaluate(str(Path(args.sr).resolve()), config["gt"], output, config["metrics"], config["device"], args.limit)
            else:
                from .evaluation import summarize
                with Path(args.csv).open(newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    names = [key for key in reader.fieldnames if key != "image"]
                    result = summarize(list(reader), names)
                result["source_sha256"] = sha256(args.csv)
                write_json(output / "summary.json", result)
            record.update(status="complete", finished=datetime.now(timezone.utc).isoformat())
            write_json(manifest, record)
            printable = {k: v for k, v in result.items() if k not in {"calls", "layers"}}
            if "activation_allocation" in printable:
                printable["activation_allocation"] = {k: v for k, v in printable["activation_allocation"].items() if k != "pairs"}
            print(json.dumps(printable, indent=2, ensure_ascii=False))
        except Exception as exc:
            record.update(status="failed", error=str(exc), finished=datetime.now(timezone.utc).isoformat())
            write_json(manifest, record)
            raise
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
