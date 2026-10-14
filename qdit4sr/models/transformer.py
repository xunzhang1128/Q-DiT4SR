# Copyright 2024 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Optional, Tuple
import torch
from torch import nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from .attention import JointTransformerBlock


class SD3Transformer2DModel(ModelMixin, ConfigMixin):
    """DiT4SR control-image Transformer, with the original checkpoint names."""

    @register_to_config
    def __init__(self, sample_size: int=128, patch_size: int=2, in_channels: int=16, num_layers: int=18, attention_head_dim: int=64, num_attention_heads: int=18, joint_attention_dim: int=4096, caption_projection_dim: int=1152, pooled_projection_dim: int=2048, out_channels: int=16, pos_embed_max_size: int=96, dual_attention_layers: Tuple[int, ...]=(), qk_norm: Optional[str]=None):
        super().__init__()
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.pos_embed = PatchEmbed(height=self.config.sample_size, width=self.config.sample_size, patch_size=self.config.patch_size, in_channels=self.config.in_channels, embed_dim=self.inner_dim, pos_embed_max_size=pos_embed_max_size)
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim)
        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.config.caption_projection_dim)
        self.transformer_blocks = nn.ModuleList([JointTransformerBlock(dim=self.inner_dim, num_attention_heads=self.config.num_attention_heads, attention_head_dim=self.config.attention_head_dim, context_pre_only=i == num_layers - 1, qk_norm=qk_norm, use_dual_attention=i in dual_attention_layers) for i in range(self.config.num_layers)])
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-06)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

    def forward(self, hidden_states, controlnet_image, encoder_hidden_states,
                pooled_projections, timestep, return_dict=True):
        if hidden_states.shape != controlnet_image.shape:
            raise ValueError("Noise and control latents must have the same shape")
        height, width = hidden_states.shape[-2:]
        hidden_states = self.pos_embed(hidden_states)
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        control = self.pos_embed(controlnet_image)
        hidden_states = torch.cat([hidden_states, control], dim=-2)
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(hidden_states, encoder_hidden_states, temb)
        hidden_states = hidden_states.chunk(2, dim=1)[0]
        hidden_states = self.proj_out(self.norm_out(hidden_states, temb))
        patch = self.config.patch_size
        hidden_states = hidden_states.reshape(hidden_states.shape[0], height // patch,
                                              width // patch, patch, patch, self.out_channels)
        output = torch.einsum("nhwpqc->nchpwq", hidden_states).reshape(
            hidden_states.shape[0], self.out_channels, height, width)
        return Transformer2DModelOutput(sample=output) if return_dict else (output,)
