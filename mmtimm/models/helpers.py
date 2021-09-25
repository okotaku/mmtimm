from timm.models.helpers import build_model_with_cfg


def mmtimm_build_model_with_cfg(*args, **kwargs):
    if kwargs.pop('features_only', False):
        features = True
        out_indices = kwargs.pop('out_indices', (0, 1, 2, 3, 4))
    else:
        features = False
    model = build_model_with_cfg(*args, **kwargs)
    if features:
        model.features_only = True
        model.out_indices = out_indices

    return model
