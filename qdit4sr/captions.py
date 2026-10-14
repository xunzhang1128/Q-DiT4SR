#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


"""LLaVA v1.5 Llama caption generation for the SR pipeline."""
import re

import torch
from torch import nn
from transformers import (AutoTokenizer, CLIPImageProcessor,
                          CLIPVisionModel, LlamaConfig, LlamaForCausalLM, LlamaModel)


_IMAGE_TOKEN = -200
_QUESTION = ("Please describe the actual objects in the image in a very detailed manner. "
             "Please do not include descriptions related to the focus and bokeh of this image. "
             "Please do not include descriptions like the background is blurred.")
_SYSTEM = ("A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.")
_FILTER = ("focus", "focal", "prominent", "close-up", "black and white", "blur", "depth", "dense", "locate", "position")


class LlavaConfig(LlamaConfig):
    model_type = "llava"


def filter_caption(text):
    parts = re.split(r"([.?!])", text)
    sentences = ["".join(parts[i:i + 2]) for i in range(0, len(parts), 2)]
    return "".join(s for s in sentences if not any(w in s.lower() for w in _FILTER)).strip()


class VisionTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.path = config.mm_vision_tower
        self.select_layer = config.mm_vision_select_layer
        if getattr(config, "mm_vision_select_feature", "patch") != "patch":
            raise ValueError("The default caption model requires CLIP patch features")

    def load(self, device):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.path)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.path).to(device=device, dtype=torch.float16)
        self.vision_tower.eval().requires_grad_(False)

    def forward(self, images):
        out = self.vision_tower(images, output_hidden_states=True)
        return out.hidden_states[self.select_layer][:, 1:].to(images.dtype)


class LlavaLlamaModel(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.vision_tower = VisionTower(config)
        if config.mm_projector_type != "mlp2x_gelu":
            raise ValueError("Only the LLaVA v1.5 mlp2x_gelu projector is supported")
        self.mm_projector = nn.Sequential(nn.Linear(config.mm_hidden_size, config.hidden_size),
                                          nn.GELU(), nn.Linear(config.hidden_size, config.hidden_size))


class LlavaLlamaForCausalLM(LlamaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @torch.no_grad()
    def caption_tokens(self, ids, images):
        if ids.shape[0] != 1 or (ids == _IMAGE_TOKEN).sum().item() != 1:
            raise ValueError("Captioning expects one image and one image token")
        index = torch.where(ids[0] == _IMAGE_TOKEN)[0].item()
        features = self.model.mm_projector(self.model.vision_tower(images))
        embeds = torch.cat([self.model.embed_tokens(ids[:, :index]), features,
                            self.model.embed_tokens(ids[:, index + 1:])], dim=1)
        return super().generate(inputs_embeds=embeds, do_sample=True, temperature=0.2,
                                top_p=0.7, num_beams=1, max_new_tokens=512, use_cache=True)


class Captioner:
    def __init__(self, model_path, clip_path, device):
        if torch.device(device).type != "cuda":
            raise ValueError("The default 8-bit LLaVA captioner requires CUDA; use cached captions for CPU evaluation")
        config = LlavaConfig.from_pretrained(model_path)
        config.mm_vision_tower = clip_path
        if getattr(config, "mm_patch_merge_type", "flat") != "flat":
            raise ValueError("Only LLaVA v1.5 flat image patches are supported")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = LlavaLlamaForCausalLM.from_pretrained(
            model_path, config=config, low_cpu_mem_usage=True, load_in_8bit=True,
            device_map={"": device}, torch_dtype=torch.float16).eval()
        if getattr(config, "mm_use_im_patch_token", True):
            self.tokenizer.add_tokens(["<im_patch>"], special_tokens=True)
        if getattr(config, "mm_use_im_start_end", False):
            self.tokenizer.add_tokens(["<im_start>", "<im_end>"], special_tokens=True)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.model.vision_tower.load(device)
        self.processor = self.model.model.vision_tower.image_processor
        marker = "<im_start><image><im_end>" if getattr(config, "mm_use_im_start_end", False) else "<image>"
        prompt = _SYSTEM + " USER: " + marker + "\n" + _QUESTION + " ASSISTANT:"
        chunks = [self.tokenizer(part).input_ids for part in prompt.split("<image>")]
        offset = int(chunks[0][0] == self.tokenizer.bos_token_id)
        ids = ([chunks[0][0]] if offset else []) + chunks[0][offset:] + [_IMAGE_TOKEN] + chunks[1][offset:]
        self.ids = torch.tensor([ids], dtype=torch.long, device=device)
        self.device = device

    @torch.inference_mode()
    def __call__(self, image):
        pixels = self.processor.preprocess(image, return_tensors="pt")["pixel_values"].to(self.device, torch.float16)
        tokens = self.model.caption_tokens(self.ids, pixels)
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
