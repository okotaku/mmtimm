import pytest
from torch.nn.modules.batchnorm import _BatchNorm


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


@pytest.mark.parametrize('backbone,arc,size',
                         [('momentumnet_resnet', 'resnet34', (1, 512, 7, 7)),
                          ('momentumnet_resnet', 'resnet34d', (1, 512, 7, 7))])
def test_timm_backbone(backbone, arc, size):
    # Test from timm
    # model = TIMMBackbone(model_name=backbone, pretrained=False, arc=arc)
    # model.train()
    # assert check_norm_state(model.modules(), True)

    # imgs = torch.randn(1, 3, 224, 224)
    # feat = model(imgs)
    # assert len(feat) == 1
    # assert feat[0].shape == torch.Size(size)

    # test reset_classifier
    # model = timm.create_model(backbone, pretrained=False, arc=arc)
    # assert not isinstance(model.fc, nn.Identity)
    # model.reset_classifier(0)
    # assert isinstance(model.fc, nn.Identity)
    pass
