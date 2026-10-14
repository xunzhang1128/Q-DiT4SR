# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
from diffusers.models.activations import GELU
from diffusers.models.attention_processor import Attention
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, SD35AdaLayerNormZeroX
from math import isqrt


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.net = nn.ModuleList([GELU(dim, dim * 4, approximate="tanh"),
                                  nn.Dropout(0.0), nn.Linear(dim * 4, dim_out)])

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class FeedForwardControl(FeedForward):
    def __init__(self, dim, dim_out):
        super().__init__(dim, dim_out)
        self.control_conv = zero_module(nn.Conv2d(dim * 4, dim * 4, 3, padding=1, groups=dim * 4))

    def forward(self, hidden_states):
        for i, module in enumerate(self.net):
            hidden_states = module(hidden_states)
            if i == 1:
                main, control = hidden_states.chunk(2, dim=1)
                batch, tokens, channels = main.shape
                side = isqrt(tokens)
                if side * side != tokens:
                    raise ValueError("DiT4SR requires square latent tiles")
                signal = self.control_conv(control.reshape(batch, side, side, channels).permute(0, 3, 1, 2))
                signal = signal.reshape(batch, channels, tokens).permute(0, 2, 1)
                hidden_states = torch.cat([main + 1.2 * signal, control], dim=1)
        return hidden_states


class JointTransformerBlock(nn.Module):

    def __init__(self, dim, num_attention_heads, attention_head_dim, context_pre_only=False, qk_norm: Optional[str]=None, use_dual_attention: bool=False):
        super().__init__()
        self.use_dual_attention = use_dual_attention
        self.context_pre_only = context_pre_only
        if use_dual_attention:
            self.norm1 = SD35AdaLayerNormZeroX(dim)
        else:
            self.norm1 = AdaLayerNormZero(dim)
        if context_pre_only:
            self.norm1_context = AdaLayerNormContinuous(dim, dim, elementwise_affine=False, eps=1e-06, bias=True, norm_type='layer_norm')
        else:
            self.norm1_context = AdaLayerNormZero(dim)
        processor = JointAttnProcessor2_0()
        self.attn = AttentionZero(query_dim=dim, cross_attention_dim=None, added_kv_proj_dim=dim, dim_head=attention_head_dim, heads=num_attention_heads, out_dim=dim, context_pre_only=context_pre_only, bias=True, processor=processor, qk_norm=qk_norm, eps=1e-06)
        if use_dual_attention:
            self.attn2 = AttentionZero(query_dim=dim, cross_attention_dim=None, added_kv_proj_dim=dim, dim_head=attention_head_dim, heads=num_attention_heads, out_dim=dim, context_pre_only=context_pre_only, bias=True, processor=processor, qk_norm=qk_norm, eps=1e-06)
        else:
            self.attn2 = None
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-06)
        self.ff = FeedForwardControl(dim=dim, dim_out=dim)
        if not context_pre_only:
            self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-06)
            self.ff_context = FeedForward(dim=dim, dim_out=dim)
        else:
            self.norm2_context = None
            self.ff_context = None

    def forward(self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor, temb: torch.FloatTensor):
        if self.use_dual_attention:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2 = self.norm1(hidden_states, emb=temb)
        else:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, emb=temb)
        attn_output, context_attn_output = self.attn(hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states)
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output
        if self.use_dual_attention:
            attn_output2 = self.attn2(hidden_states=norm_hidden_states2)
            attn_output2 = gate_msa2.unsqueeze(1) * attn_output2
            hidden_states = hidden_states + attn_output2
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output
            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        return (encoder_hidden_states, hidden_states)

class AttentionZero(Attention):

    def __init__(self, query_dim, cross_attention_dim, added_kv_proj_dim, dim_head, heads, out_dim, context_pre_only, bias, processor, qk_norm, eps):
        super(AttentionZero, self).__init__(query_dim=query_dim, cross_attention_dim=cross_attention_dim, added_kv_proj_dim=added_kv_proj_dim, dim_head=dim_head, heads=heads, out_dim=out_dim, context_pre_only=context_pre_only, bias=bias, processor=processor, qk_norm=qk_norm, eps=1e-06)
        self.to_q_control = zero_module(nn.Linear(self.query_dim, self.inner_dim, bias=self.use_bias))
        self.to_k_control = zero_module(nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=self.use_bias))
        self.to_v_control = zero_module(nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=self.use_bias))
        self.to_out_control = nn.Linear(self.inner_dim, self.out_dim, bias=True)
        self.to_out_control.weight.data.copy_(self.to_out[0].weight.data)
        self.to_out_control.bias.data.copy_(self.to_out[0].bias.data)

class JointAttnProcessor2_0:

    def __init__(self):
        if not hasattr(F, 'scaled_dot_product_attention'):
            raise ImportError('AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.')

    def __call__(self, attn: Attention, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor=None, attention_mask: Optional[torch.FloatTensor]=None, *args, **kwargs) -> torch.FloatTensor:
        residual = hidden_states
        batch_size = hidden_states.shape[0]
        hidden_states, hidden_states_control = hidden_states.chunk(2, dim=1)
        hidden_states_control_res = hidden_states_control
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        query_control = attn.to_q_control(attn.to_q(hidden_states_control))
        key_control = attn.to_k_control(attn.to_k(hidden_states_control))
        value_control = attn.to_v_control(attn.to_v(hidden_states_control))
        query_control = query_control.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key_control = key_control.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_control = value_control.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
            query_control = attn.norm_q(query_control)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
            key_control = attn.norm_k(key)
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
            query = torch.cat([query, query_control, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, key_control, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, value_control, encoder_hidden_states_value_proj], dim=2)
        else:
            query = torch.cat([query, query_control], dim=2)
            key = torch.cat([key, key_control], dim=2)
            value = torch.cat([value, value_control], dim=2)
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (hidden_states[:, :residual.shape[1]], hidden_states[:, residual.shape[1]:])
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        hidden_states, hidden_states_control = hidden_states.chunk(2, dim=1)
        hidden_states_control = hidden_states_control + hidden_states_control_res
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states_control = attn.to_out_control(hidden_states_control)
        hidden_states = torch.cat([hidden_states, hidden_states_control], dim=1)
        if encoder_hidden_states is not None:
            return (hidden_states, encoder_hidden_states)
        else:
            return hidden_states
