"""Weight statistics, actual Linear work, and quantization parameter accounting."""
import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from tqdm import tqdm

from .models import SD3Transformer2DModel
from .quantization.linear import target_linears
from .quantization.hadamard import transform
from .quantization.blr import block_config
from .quantization.allocation import summarize_bits
from .quantization.activation import load_schedule, summarize_schedule
from .io import read_layer_table, sha256


def model_skeleton(path):
    config = json.loads((Path(path) / "config.json").read_text())
    # Positional buffer extent does not affect Linear parameter shapes or calls.
    with init_empty_weights(include_buffers=True):
        model = SD3Transformer2DModel.from_config(config, pos_embed_max_size=32)
    return model.to("meta").eval()


def load_transformer(path):
    model, info = SD3Transformer2DModel.from_pretrained(
        path, torch_dtype=torch.float32, low_cpu_mem_usage=True, output_loading_info=True)
    problems = {key: info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if info.get(key)}
    if problems:
        raise ValueError("Transformer checkpoint does not match architecture: " + str(problems))
    return model.eval().requires_grad_(False)


@torch.no_grad()
def weight_statistics(path, device, seed, layer_order=None):
    layers = target_linears(model_skeleton(path))
    # Allocation preserves CSV order, which also determines export's RNG traversal.
    if layer_order is not None:
        if len(layer_order) != len(layers) or set(layer_order) != set(layers):
            raise ValueError("Statistics layer order does not match the Transformer")
        layers = {name: layers[name] for name in layer_order}
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = []
    checkpoint = Path(path) / "diffusion_pytorch_model.safetensors"
    with safe_open(checkpoint, framework="pt", device="cpu") as weights:
        for name, layer in tqdm(layers.items(), total=len(layers), desc="Weight statistics", mininterval=2):
            weight = weights.get_tensor(name + ".weight").to(device=device, dtype=torch.float32)
            if tuple(weight.shape) != (layer.out_features, layer.in_features):
                raise ValueError("Checkpoint shape mismatch for " + name)
            signs = (torch.randint(0, 2, (layer.in_features,), generator=generator).float() * 2 - 1).to(device)
            transformed = transform(weight * signs)
            rows.append(dict(layer_name=name, variance=transformed.var(dim=-1, unbiased=False).mean().item(),
                             n_params=weight.numel()))
    return rows, dict(seed=seed, definition="FP32 signed-Hadamard weight; population variance per output row, averaged across rows; before SVD subtraction",
                      checkpoint_sha256=sha256(checkpoint))


class ActivationVarianceRecorder:
    """Reference post-Hadamard token variance, including CPU reduction order."""
    def __init__(self):
        self.accum, self.handles = {}, []
        self.start_image()

    def start_image(self):
        self.step_idx, self.last_t, self.timestep = -1, None, None

    def update_step(self, timestep):
        value = int(timestep.flatten()[0].item()) if torch.is_tensor(timestep) else int(timestep)
        if self.last_t is None or value != self.last_t:
            self.step_idx += 1
            self.last_t = value
        self.timestep = value

    def record(self, name, tensor):
        x = tensor.detach().float()
        x = x.reshape(-1, x.shape[-1]).cpu()
        variance = x.var(dim=-1, unbiased=False)
        mean, tokens = float(variance.mean().item()), variance.numel()
        entry = self.accum.setdefault((self.step_idx, self.timestep, name), [0.0, 0, 0])
        entry[0] += mean * tokens
        entry[1] += tokens
        entry[2] += 1

    def attach(self, transformer):
        from .quantization.linear import MixedLinear
        def hook(module, args, kwargs):
            self.update_step(kwargs["timestep"])
        self.handles.append(transformer.register_forward_pre_hook(hook, with_kwargs=True))
        self.layers = []
        for name, module in transformer.named_modules():
            if isinstance(module, MixedLinear):
                module.post_hadamard_tap = lambda x, name=name: self.record(name, x)
                self.layers.append(module)

    def detach(self):
        for handle in self.handles:
            handle.remove()
        for module in self.layers:
            module.post_hadamard_tap = None
        self.handles, self.layers = [], []

    def rows(self):
        return [dict(step_idx=step, timestep=timestep, layer_name=name,
                     mean_token_var=values[0] / values[1], num_tokens=values[1], num_records=values[2])
                for (step, timestep, name), values in sorted(self.accum.items())]


@torch.no_grad()
def profile(path, bit_path, quant_config, image_size=512, steps=40):
    if image_size % 16 or image_size <= 0 or image_size > 512:
        raise ValueError("Profile a square latent tile: image_size must be a multiple of 16 up to 512")
    model = model_skeleton(path)
    targets = target_linears(model)
    rows = read_layer_table(bit_path, with_bits=True)
    bits = {r["layer_name"]: r for r in rows}
    if set(bits) != set(targets) or any(bits[n]["n_params"] != m.weight.numel() for n, m in targets.items()):
        raise ValueError("Bit map does not match Transformer")
    calls, handles = [], []
    svd_parameters, blr_parameters = 0, 0
    for name, module in targets.items():
        rank = min(quant_config["svd_rank"], module.in_features, module.out_features)
        svd_parameters += rank * (module.in_features + module.out_features)
        if quant_config["blr_rank_budget"]:
            bo, bi, r = block_config(module.in_features, module.out_features, quant_config["blr_rank_budget"])
            blr_parameters += (module.out_features // bo) * (module.in_features // bi) * r * (bo + bi)
    def hook(name):
        def record(module, inputs, output):
            tokens = inputs[0].numel() // module.in_features
            calls.append(dict(layer_name=name, tokens=tokens, in_features=module.in_features,
                              out_features=module.out_features, macs=tokens * module.weight.numel()))
        return record
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(hook(name)))
    size = image_size // 8
    model(hidden_states=torch.empty(2, 16, size, size, device="meta"),
          controlnet_image=torch.empty(2, 16, size, size, device="meta"),
          encoder_hidden_states=torch.empty(2, 333, model.config.joint_attention_dim, device="meta"),
          pooled_projections=torch.empty(2, model.config.pooled_projection_dim, device="meta"),
          timestep=torch.empty(2, device="meta"))
    for handle in handles:
        handle.remove()
    called = {call["layer_name"] for call in calls if call["layer_name"] in targets}
    dense_mac = sum(c["macs"] for c in calls if c["layer_name"] in targets)
    other_mac = sum(c["macs"] for c in calls if c["layer_name"] not in targets)
    svd_mac = sum(c["tokens"] * min(quant_config["svd_rank"], c["in_features"], c["out_features"]) *
                  (c["in_features"] + c["out_features"]) for c in calls if c["layer_name"] in targets)
    # BLR.forward reconstructs a dense matrix; count the executed dense F.linear
    # and its block reconstruction separately, rather than reporting factored cost.
    blr_mac = dense_mac if quant_config["blr_rank_budget"] else 0
    reconstruction = 0
    if quant_config["blr_rank_budget"]:
        for call in calls:
            if call["layer_name"] in targets:
                _, _, rank = block_config(call["in_features"], call["out_features"], quant_config["blr_rank_budget"])
                reconstruction += call["in_features"] * call["out_features"] * rank
    actual = dense_mac + other_mac + svd_mac + blr_mac + reconstruction
    original_parameters = sum(p.numel() for p in model.parameters())
    result = dict(image_size=image_size, batch=2, text_tokens=333, steps=steps, bit_allocation=summarize_bits(rows),
                  original_transformer_parameters=original_parameters, svd_parameters=svd_parameters,
                  blr_parameters=blr_parameters, quantized_transformer_parameters=original_parameters + svd_parameters + blr_parameters,
                  called_quantized_layers=len(called), called_quantized_parameters=sum(bits[n]["n_params"] for n in called),
                  called_parameter_weighted_bits=sum(bits[n]["n_params"] * bits[n]["bits_assigned"] for n in called) / sum(bits[n]["n_params"] for n in called),
                  linear_calls=len(calls), quantized_linear_calls=sum(c["layer_name"] in targets for c in calls),
                  dense_quant_macs=dense_mac, svd_macs=svd_mac, blr_dense_macs=blr_mac,
                  blr_reconstruction_macs=reconstruction, other_linear_macs=other_mac,
                  executed_linear_macs_per_step=actual, executed_linear_flops_per_step=2 * actual,
                  executed_linear_flops_all_steps=2 * actual * steps,
                  scope="Transformer Linear branches only; excludes attention, convolution, normalization, Hadamard, quantizer, text encoders and VAE. FP16 simulated quantization; these are operation counts, not integer-kernel speedups.",
                  calls=calls)
    schedule = load_schedule(quant_config["activation_schedule"], steps)
    if schedule and set(schedule) != called:
        raise ValueError("Activation schedule does not match the executed quantized layers")
    result["activation_allocation"] = summarize_schedule(schedule, rows, steps) if schedule else dict(mode="uniform", a_bits=quant_config["a_bits"])
    return result
