# --------------------------------------------------------
# Model from official source: https://github.com/michaelsdr/momentumnet
# --------------------------------------------------------
from momentumnet import transform_to_momentumnet
from timm.models import resnet
from timm.models.registry import register_model


@register_model
def momentumnet_resnet(arc,
                       pretrained=False,
                       gamma=0.9,
                       use_backprop=False,
                       **kwargs):
    model = getattr(resnet, arc)(pretrained, **kwargs)
    model = transform_to_momentumnet(model,
                                     gamma=gamma,
                                     use_backprop=use_backprop)

    return model
