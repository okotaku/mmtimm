# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# --------------------------------------------------------
# Model from official source: https://github.com/microsoft/Cream
# --------------------------------------------------------
from functools import partial

import torch
import torch.nn as nn
from timm.models.helpers import build_model_with_cfg
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.models.vision_transformer_hybrid import HybridEmbed

from ..layers import build_rpe, get_rpe_config

default_cfgs = {
    'deit_small_patch16_224_ctx_product_50_shared_qkv': {
        'url':
        'https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/deit_small_patch16_224_ctx_product_50_shared_qkv.pth'  # noqa
    },
    'deit_base_patch16_224_ctx_product_50_shared_qkv': {
        'url':
        'https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/deit_base_patch16_224_ctx_product_50_shared_qkv.pth'  # noqa
    }
}


class RPEAttention(nn.Module):
    """Attention with image relative position encoding."""
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 rpe_config=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version,
        # can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # image relative position encoding
        self.rpe_q, self.rpe_k, self.rpe_v = \
            build_rpe(rpe_config,
                      head_dim=head_dim,
                      num_heads=num_heads)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q *= self.scale

        attn = (q @ k.transpose(-2, -1))

        # image relative position on keys
        if self.rpe_k is not None:
            attn += self.rpe_k(q)

        # image relative position on queries
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale).transpose(2, 3)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v

        # image relative position on values
        if self.rpe_v is not None:
            out += self.rpe_v(attn)

        x = out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RPEBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 rpe_config=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RPEAttention(dim,
                                 num_heads=num_heads,
                                 qkv_bias=qkv_bias,
                                 qk_scale=qk_scale,
                                 attn_drop=attn_drop,
                                 proj_drop=drop,
                                 rpe_config=rpe_config)
        # NOTE: drop path for stochastic depth,
        # we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage and
    image relative position encoding."""
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 hybrid_backbone=None,
                 norm_layer=nn.LayerNorm,
                 rpe_config=None):
        super().__init__()
        self.num_classes = num_classes
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(hybrid_backbone,
                                           img_size=img_size,
                                           in_chans=in_chans,
                                           embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(img_size=img_size,
                                          patch_size=patch_size,
                                          in_chans=in_chans,
                                          embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            RPEBlock(dim=embed_dim,
                     num_heads=num_heads,
                     mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias,
                     qk_scale=qk_scale,
                     drop=drop_rate,
                     attn_drop=attn_drop_rate,
                     drop_path=dpr[i],
                     norm_layer=norm_layer,
                     rpe_config=rpe_config) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a
        # pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _filter_fn(state_dict):
    state_dict = state_dict['model']
    return state_dict


def _irpe(arch, pretrained, **kwargs):
    model = build_model_with_cfg(VisionTransformer,
                                 arch,
                                 pretrained=pretrained,
                                 default_cfg=default_cfgs[arch],
                                 pretrained_filter_fn=_filter_fn,
                                 **kwargs)
    return model


@register_model
def deit_small_patch16_224_ctx_product_50_shared_qkv(pretrained=False,
                                                     **kwargs):
    rpe_config = get_rpe_config(
        ratio=1.9,
        method='product',
        mode='ctx',
        shared_head=True,
        skip=1,
        rpe_on='qkv',
    )
    return _irpe('deit_small_patch16_224_ctx_product_50_shared_qkv',
                 pretrained=pretrained,
                 rpe_config=rpe_config,
                 patch_size=16,
                 embed_dim=384,
                 depth=12,
                 num_heads=6,
                 mlp_ratio=4,
                 qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 **kwargs)


@register_model
def deit_base_patch16_224_ctx_product_50_shared_qkv(pretrained=False,
                                                    **kwargs):
    rpe_config = get_rpe_config(
        ratio=1.9,
        method='product',
        mode='ctx',
        shared_head=True,
        skip=1,
        rpe_on='qkv',
    )
    return _irpe('deit_base_patch16_224_ctx_product_50_shared_qkv',
                 pretrained=pretrained,
                 rpe_config=rpe_config,
                 patch_size=16,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 **kwargs)
