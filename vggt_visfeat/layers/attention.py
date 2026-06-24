# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

from torch import Tensor
from torch import nn
import torch.nn.functional as F
import torch
from vggt_visfeat.layers.attention_utils import per_image_confidence_minmem

XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None, save_attn=None, topk=None, att_mask=None, rgb_mask=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        if self.fused_attn:
            if att_mask is not None:
                lam = 1  # 控制门控强度
                eps = 1e-6
                M_final = att_mask * rgb_mask
                attn_bias = lam * torch.log(M_final.clamp_min(eps))  # (B, Nk)
                attn_mask = attn_bias[:, None, None, :].to(dtype=q.dtype, device=q.device)  # (B,1,1,1374)
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,  # 注意力 logits 偏置
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                    is_causal=False
                )
            else:
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)  # torch.Size([6, 16, 1374, 64])
            # if save_attn:
            #     scale = (self.head_dim ** -0.5)
            #     logits = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B,H,N,N]
            #     attn = torch.softmax(logits, dim=-1)
            #     self.last_attn = attn.detach()
            if save_attn:
                scale = self.head_dim ** -0.5
                logits_clean = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, Nq, Nk]
                # Unmasked attention: captures intrinsic patch distinctiveness (for sharpness)
                self.last_attn_unmasked = torch.softmax(logits_clean, dim=-1).detach()
                # Masked attention: reflects persp/rgb quality bias (for recv and sym)
                if att_mask is not None:
                    M_final = att_mask * rgb_mask
                    save_bias = torch.log(M_final.clamp_min(1e-6))
                    logits_masked = logits_clean.cuda() + save_bias[:, None, None, :].cuda()
                    self.last_attn = torch.softmax(logits_masked, dim=-1).detach()
                else:
                    self.last_attn = self.last_attn_unmasked

        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            if save_attn:
                self.last_attn = attn.detach()
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None,save_attn=None,  topk=None, att_mask=None, rgb_mask=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
