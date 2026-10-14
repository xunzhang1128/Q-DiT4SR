"""SD3 text/VAE components with the DiT4SR control-image denoising loop."""
import math

import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline


def _tile_starts(length, side, overlap):
    if length <= side:
        return [0]
    return list(range(0, length - side, side - overlap)) + [length - side]


def load_pipeline(base, transformer):
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5Tokenizer
    options = dict(torch_dtype=torch.float16, local_files_only=True)
    return DiT4SRPipeline(
        transformer=transformer,
        vae=AutoencoderKL.from_pretrained(base, subfolder="vae", **options),
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(base, subfolder="scheduler", local_files_only=True),
        text_encoder=CLIPTextModelWithProjection.from_pretrained(base, subfolder="text_encoder", **options),
        text_encoder_2=CLIPTextModelWithProjection.from_pretrained(base, subfolder="text_encoder_2", **options),
        text_encoder_3=T5EncoderModel.from_pretrained(base, subfolder="text_encoder_3", **options),
        tokenizer=CLIPTokenizer.from_pretrained(base, subfolder="tokenizer", local_files_only=True),
        tokenizer_2=CLIPTokenizer.from_pretrained(base, subfolder="tokenizer_2", local_files_only=True),
        # The reference uses SentencePiece, avoiding newer fast-tokenizer file formats.
        tokenizer_3=T5Tokenizer.from_pretrained(base, subfolder="tokenizer_3", local_files_only=True),
    )


class DiT4SRPipeline(StableDiffusion3Pipeline):
    @torch.no_grad()
    def _predict(self, latent, control, timestep, prompt, pooled, side, overlap):
        height, width = latent.shape[-2:]
        if height * width <= side * side:
            return self.transformer(hidden_states=latent, controlnet_image=control,
                                    timestep=timestep, encoder_hidden_states=prompt,
                                    pooled_projections=pooled, return_dict=False)[0]
        # Retain legacy tile order, FP64 Gaussian products and FP32 accumulation.
        side = min(side, height, width)
        ys, xs = _tile_starts(height, side, overlap), _tile_starts(width, side, overlap)
        x = [np.exp(-(i - (side - 1) / 2) ** 2 / (side * side) / .02) / math.sqrt(.02 * math.pi) for i in range(side)]
        y = [np.exp(-(i - side / 2) ** 2 / (side * side) / .02) / math.sqrt(.02 * math.pi) for i in range(side)]
        weights = torch.tensor(np.outer(y, x), device=latent.device)
        prediction = torch.zeros_like(latent, dtype=torch.float32)
        normalization = torch.zeros_like(latent, dtype=torch.float32)
        for left in xs:
            for top in ys:
                region = (..., slice(top, top + side), slice(left, left + side))
                tile = self.transformer(hidden_states=latent[region], controlnet_image=control[region],
                                        timestep=timestep, encoder_hidden_states=prompt,
                                        pooled_projections=pooled, return_dict=False)[0]
                prediction[region] += tile * weights
                normalization[region] += weights
        return prediction / normalization

    @torch.no_grad()
    def __call__(self, image, prompt, negative_prompt, generator, steps=40, guidance_scale=8.0,
                 tile_size=64, tile_overlap=24, vae_input_layout="contiguous"):
        if not 0 <= tile_overlap < tile_size:
            raise ValueError("Latent tile overlap must be smaller than tile size")
        width, height = image.size
        if width % 8 or height % 8:
            raise ValueError("Control image dimensions must be multiples of 8")
        device = self._execution_device
        self._guidance_scale = guidance_scale
        self._clip_skip = None
        self._joint_attention_kwargs = None
        self._interrupt = False
        cfg = guidance_scale > 1
        embeds, negative, pooled, negative_pooled = self.encode_prompt(
            prompt=prompt, prompt_2=None, prompt_3=None, device=device,
            num_images_per_prompt=1, do_classifier_free_guidance=cfg,
            negative_prompt=negative_prompt, max_sequence_length=256)
        if cfg:
            embeds = torch.cat([negative, embeds])
            pooled = torch.cat([negative_pooled, pooled])
        self.scheduler.set_timesteps(steps, device=device)
        self._num_timesteps = len(self.scheduler.timesteps)
        # Input strides select the FP16 VAE convolution kernels used by each saved run.
        memory_format = torch.channels_last if vae_input_layout == "channels_last" else torch.contiguous_format
        condition = self.image_processor.preprocess(image, height=height, width=width).contiguous(
            memory_format=memory_format).to(device, self.vae.dtype)
        if cfg:
            condition = torch.cat([condition, condition])
        # The legacy VAE samples from the global RNG, independently of noise generator.
        control = self.vae.encode(condition).latent_dist.sample()
        control = (control - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        latents = self.prepare_latents(1, self.transformer.config.in_channels, height, width,
                                       embeds.dtype, device, generator)
        for t in self.progress_bar(self.scheduler.timesteps):
            model_input = torch.cat([latents, latents]) if cfg else latents
            noise = self._predict(model_input, control, t.expand(model_input.shape[0]),
                                  embeds, pooled, tile_size, tile_overlap)
            if cfg:
                unconditional, conditional = noise.chunk(2)
                noise = unconditional + guidance_scale * (conditional - unconditional)
            latents = self.scheduler.step(noise, t, latents, return_dict=False)[0]
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        decoded = self.vae.decode(latents, return_dict=False)[0]
        if not torch.isfinite(decoded).all():
            raise ValueError("VAE output contains nonfinite values")
        return self.image_processor.postprocess(decoded, output_type="pil")[0]


def adain_color_fix(target, reference):
    """Match RGB mean/std, using the original sample variance and epsilon."""
    from torchvision.transforms.functional import to_tensor, to_pil_image
    target, reference = to_tensor(target).unsqueeze(0), to_tensor(reference).unsqueeze(0)
    def moments(x):
        flattened = x.reshape(x.shape[0], x.shape[1], -1)
        mean = flattened.mean(dim=2).view(x.shape[0], x.shape[1], 1, 1)
        std = (flattened.var(dim=2) + 1e-5).sqrt().view_as(mean)
        return mean, std
    mean_t, std_t = moments(target)
    mean_r, std_r = moments(reference)
    result = (target - mean_t) / std_t * std_r + mean_r
    return to_pil_image(result.squeeze(0).clamp(0, 1))
