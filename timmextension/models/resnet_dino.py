# --------------------------------------------------------
# Model from official source: https://github.com/facebookresearch/dino
# --------------------------------------------------------
from timm.models.helpers import build_model_with_cfg
from timm.models.registry import register_model
from timm.models.resnet import Bottleneck, ResNet, _cfg

default_cfgs = {
    # ResNet and Wide ResNet
    'resnet50_dino':
    _cfg(
        url=
        'https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth',  # noqa
        interpolation='bicubic',
        crop_pct=0.95),
}


def _filter_fn(state_dict):
    state_dict['fc.weight'] = None
    state_dict['fc.bias'] = None
    return state_dict


def _create_resnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(ResNet,
                                variant,
                                pretrained,
                                default_cfg=default_cfgs[variant],
                                pretrained_filter_fn=_filter_fn,
                                **kwargs)


@register_model
def resnet50_dino(pretrained=False, num_classes=0, **kwargs):
    """Constructs a ResNet-50 model."""
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    return _create_resnet('resnet50_dino',
                          pretrained,
                          num_classes=num_classes,
                          **model_args)
