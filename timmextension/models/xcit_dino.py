# --------------------------------------------------------
# Model from official source: https://github.com/facebookresearch/dino
# --------------------------------------------------------
# Todo: weight for timm update
from timm.models.helpers import build_model_with_cfg
from timm.models.registry import register_model
from timm.models.xcit import XCiT, _cfg, checkpoint_filter_fn

default_cfgs = {
    'xcit_small_12_p16_dino':
    _cfg(
        url=
        'https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth'  # noqa
    ),
    'xcit_small_12_p8_dino':
    _cfg(
        url=
        'https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth'  # noqa
    ),
    'xcit_medium_24_p16_dino':
    _cfg(
        url=
        'https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth'  # noqa
    ),
    'xcit_medium_24_p8_dino':
    _cfg(
        url=
        'https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth'  # noqa
    ),
}


def _checkpoint_filter_fn(state_dict, model):
    state_dict = checkpoint_filter_fn(state_dict, model)
    state_dict['head.weight'] = None
    state_dict['head.bias'] = None
    return state_dict


def _create_xcit(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    model = build_model_with_cfg(XCiT,
                                 variant,
                                 pretrained,
                                 default_cfg=default_cfg,
                                 pretrained_filter_fn=_checkpoint_filter_fn,
                                 **kwargs)
    return model


@register_model
def xcit_small_12_p16_dino(pretrained=False, num_classes=0, **kwargs):
    model_kwargs = dict(num_classes=num_classes,
                        patch_size=16,
                        embed_dim=384,
                        depth=12,
                        num_heads=8,
                        eta=1.0,
                        tokens_norm=True,
                        **kwargs)
    model = _create_xcit('xcit_small_12_p16_dino',
                         pretrained=pretrained,
                         **model_kwargs)
    return model


@register_model
def xcit_small_12_p8_dino(pretrained=False, num_classes=0, **kwargs):
    model_kwargs = dict(num_classes=num_classes,
                        patch_size=8,
                        embed_dim=384,
                        depth=12,
                        num_heads=8,
                        eta=1.0,
                        tokens_norm=True,
                        **kwargs)
    model = _create_xcit('xcit_small_12_p8_dino',
                         pretrained=pretrained,
                         **model_kwargs)
    return model


@register_model
def xcit_medium_24_p16_dino(pretrained=False, num_classes=0, **kwargs):
    model_kwargs = dict(num_classes=num_classes,
                        patch_size=16,
                        embed_dim=512,
                        depth=24,
                        num_heads=8,
                        eta=1e-5,
                        tokens_norm=True,
                        **kwargs)
    model = _create_xcit('xcit_medium_24_p16_dino',
                         pretrained=pretrained,
                         **model_kwargs)
    return model


@register_model
def xcit_medium_24_p8_dino(pretrained=False, num_classes=0, **kwargs):
    model_kwargs = dict(num_classes=num_classes,
                        patch_size=8,
                        embed_dim=512,
                        depth=24,
                        num_heads=8,
                        eta=1e-5,
                        tokens_norm=True,
                        **kwargs)
    model = _create_xcit('xcit_medium_24_p8_dino',
                         pretrained=pretrained,
                         **model_kwargs)
    return model
