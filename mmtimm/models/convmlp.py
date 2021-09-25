# --------------------------------------------------------
# Model from official source: https://github.com/SHI-Labs/Convolutional-MLPs
# --------------------------------------------------------
import torch
import torch.nn as nn
from timm.models.features import FeatureInfo
from timm.models.registry import register_model
from torch.nn import (GELU, BatchNorm2d, Conv2d, Identity, LayerNorm, Linear,
                      Module, ModuleList, ReLU, Sequential)

from .helpers import mmtimm_build_model_with_cfg

__all__ = ['ConvMLP', 'convmlp_s', 'convmlp_m', 'convmlp_l']

default_cfgs = {
    'convmlp_s': {
        'url':
        'http://ix.cs.uoregon.edu/~alih/conv-mlp/checkpoints/convmlp_s_imagenet.pth'  # noqa
    },
    'convmlp_m': {
        'url':
        'http://ix.cs.uoregon.edu/~alih/conv-mlp/checkpoints/convmlp_m_imagenet.pth'  # noqa
    },
    'convmlp_l': {
        'url':
        'http://ix.cs.uoregon.edu/~alih/conv-mlp/checkpoints/convmlp_l_imagenet.pth'  # noqa
    },
}


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Obtained from: github.com:rwightman/pytorch-image-models Drop paths
    (Stochastic Depth) per sample (when applied in main path of residual
    blocks).

    This is the same as the DropConnect impl I created for EfficientNet,
     etc networks, however,
    the original name is misleading as
     'Drop Connect' is a different form of dropout in a separate paper...
    See discussion:
     https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956
     ... I've opted for
    changing the layer and argument names to 'drop path'
    rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (
        x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Obtained from: github.com:rwightman/pytorch-image-models Drop paths
    (Stochastic Depth) per sample  (when applied in main path of residual
    blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvStage(Module):
    def __init__(self,
                 num_blocks=2,
                 embedding_dim_in=64,
                 hidden_dim=128,
                 embedding_dim_out=128):
        super(ConvStage, self).__init__()
        self.conv_blocks = ModuleList()
        for i in range(num_blocks):
            block = Sequential(
                Conv2d(embedding_dim_in,
                       hidden_dim,
                       kernel_size=(1, 1),
                       stride=(1, 1),
                       padding=(0, 0),
                       bias=False), BatchNorm2d(hidden_dim),
                ReLU(inplace=True),
                Conv2d(hidden_dim,
                       hidden_dim,
                       kernel_size=(3, 3),
                       stride=(1, 1),
                       padding=(1, 1),
                       bias=False), BatchNorm2d(hidden_dim),
                ReLU(inplace=True),
                Conv2d(hidden_dim,
                       embedding_dim_in,
                       kernel_size=(1, 1),
                       stride=(1, 1),
                       padding=(0, 0),
                       bias=False), BatchNorm2d(embedding_dim_in),
                ReLU(inplace=True))
            self.conv_blocks.append(block)
        self.downsample = Conv2d(embedding_dim_in,
                                 embedding_dim_out,
                                 kernel_size=(3, 3),
                                 stride=(2, 2),
                                 padding=(1, 1))

    def forward(self, x):
        for block in self.conv_blocks:
            x = x + block(x)
        return self.downsample(x)


class Mlp(Module):
    def __init__(self,
                 embedding_dim_in,
                 hidden_dim=None,
                 embedding_dim_out=None,
                 activation=GELU):
        super().__init__()
        hidden_dim = hidden_dim or embedding_dim_in
        embedding_dim_out = embedding_dim_out or embedding_dim_in
        self.fc1 = Linear(embedding_dim_in, hidden_dim)
        self.act = activation()
        self.fc2 = Linear(hidden_dim, embedding_dim_out)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class ConvMLPStage(Module):
    def __init__(self,
                 embedding_dim,
                 dim_feedforward=2048,
                 stochastic_depth_rate=0.1):
        super(ConvMLPStage, self).__init__()
        self.norm1 = LayerNorm(embedding_dim)
        self.channel_mlp1 = Mlp(embedding_dim_in=embedding_dim,
                                hidden_dim=dim_feedforward)
        self.norm2 = LayerNorm(embedding_dim)
        self.connect = Conv2d(embedding_dim,
                              embedding_dim,
                              kernel_size=(3, 3),
                              stride=(1, 1),
                              padding=(1, 1),
                              groups=embedding_dim,
                              bias=False)
        self.connect_norm = LayerNorm(embedding_dim)
        self.channel_mlp2 = Mlp(embedding_dim_in=embedding_dim,
                                hidden_dim=dim_feedforward)
        self.drop_path = DropPath(
            stochastic_depth_rate) if stochastic_depth_rate > 0 else Identity(
            )  # noqa

    def forward(self, src):
        src = src + self.drop_path(self.channel_mlp1(self.norm1(src)))
        src = self.connect(self.connect_norm(src).permute(0, 3, 1, 2)).permute(
            0, 2, 3, 1)
        src = src + self.drop_path(self.channel_mlp2(self.norm2(src)))
        return src


class ConvDownsample(Module):
    def __init__(self, embedding_dim_in, embedding_dim_out):
        super().__init__()
        self.downsample = Conv2d(embedding_dim_in,
                                 embedding_dim_out,
                                 kernel_size=(3, 3),
                                 stride=(2, 2),
                                 padding=(1, 1))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.downsample(x)
        return x.permute(0, 2, 3, 1)


class BasicStage(Module):
    def __init__(self,
                 num_blocks,
                 embedding_dims,
                 mlp_ratio=1,
                 stochastic_depth_rate=0.1,
                 downsample=True):
        super(BasicStage, self).__init__()
        self.blocks = ModuleList()
        dpr = [
            x.item()
            for x in torch.linspace(0, stochastic_depth_rate, num_blocks)
        ]
        for i in range(num_blocks):
            block = ConvMLPStage(
                embedding_dim=embedding_dims[0],
                dim_feedforward=int(embedding_dims[0] * mlp_ratio),
                stochastic_depth_rate=dpr[i],
            )
            self.blocks.append(block)

        self.downsample_mlp = ConvDownsample(
            embedding_dims[0],
            embedding_dims[1]) if downsample else Identity()

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.downsample_mlp(x)
        return x


class ConvTokenizer(nn.Module):
    def __init__(self, embedding_dim=64):
        super(ConvTokenizer, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3,
                      embedding_dim // 2,
                      kernel_size=(3, 3),
                      stride=(2, 2),
                      padding=(1, 1),
                      bias=False), nn.BatchNorm2d(embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim // 2,
                      embedding_dim // 2,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1),
                      bias=False), nn.BatchNorm2d(embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim // 2,
                      embedding_dim,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1),
                      bias=False), nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3),
                         stride=(2, 2),
                         padding=(1, 1),
                         dilation=(1, 1)))

    def forward(self, x):
        return self.block(x)


class ConvMLP(nn.Module):
    def __init__(self,
                 blocks,
                 dims,
                 mlp_ratios,
                 channels=64,
                 n_conv_blocks=3,
                 classifier_head=True,
                 num_classes=1000,
                 features_only=False,
                 out_indices=(0, 1, 2, 3),
                 *args,
                 **kwargs):
        super(ConvMLP, self).__init__()
        assert len(blocks) == len(mlp_ratios) == len(mlp_ratios), \
            'blocks, dims and mlp_ratios must agree in size,' \
            f' {len(blocks)}, {len(dims)} and {len(mlp_ratios)} passed.'

        self.features_only = features_only
        self.out_indices = out_indices
        self.feature_info = FeatureInfo([
            {
                'num_chs': dims[0],
                'reduction': 8,
                'module': None
            },
            {
                'num_chs': dims[1],
                'reduction': 16,
                'module': None
            },
            {
                'num_chs': dims[2],
                'reduction': 32,
                'module': None
            },
            {
                'num_chs': dims[2],
                'reduction': 32,
                'module': None
            },
        ], out_indices)
        self.tokenizer = ConvTokenizer(embedding_dim=channels)
        self.conv_stages = ConvStage(n_conv_blocks,
                                     embedding_dim_in=channels,
                                     hidden_dim=dims[0],
                                     embedding_dim_out=dims[0])

        self.stages = nn.ModuleList()
        for i in range(0, len(blocks)):
            stage = BasicStage(num_blocks=blocks[i],
                               embedding_dims=dims[i:i + 2],
                               mlp_ratio=mlp_ratios[i],
                               stochastic_depth_rate=0.1,
                               downsample=(i + 1 < len(blocks)))
            self.stages.append(stage)
        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(
            dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self.init_weight)

    def reset_classifier(self, x):
        self.head = nn.Identity()

    def forward_features(self, x):
        x = self.tokenizer(x)
        x = self.conv_stages(x)
        if self.features_only:
            features = []
            if 0 in self.out_indices:
                features.append(x)
        x = x.permute(0, 2, 3, 1)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if self.features_only:
                if i + 1 in self.out_indices:
                    features.append(x.permute(0, 3, 1, 2))
        B, _, _, C = x.shape
        x = x.reshape(B, -1, C)
        x = self.norm(x)
        x = x.mean(dim=1)
        if self.features_only:
            return features
        else:
            return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.features_only:
            return x
        x = self.head(x)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, (nn.Linear, nn.Conv1d)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)


def _convmlp(arch, pretrained, classifier_head, blocks, dims, mlp_ratios,
             *args, **kwargs):
    model = mmtimm_build_model_with_cfg(ConvMLP,
                                        arch,
                                        pretrained=pretrained,
                                        default_cfg=default_cfgs[arch],
                                        blocks=blocks,
                                        dims=dims,
                                        mlp_ratios=mlp_ratios,
                                        classifier_head=classifier_head,
                                        *args,
                                        **kwargs)
    return model


@register_model
def convmlp_s(pretrained=False, classifier_head=True, *args, **kwargs):
    return _convmlp('convmlp_s',
                    pretrained=pretrained,
                    blocks=[2, 4, 2],
                    mlp_ratios=[2, 2, 2],
                    dims=[128, 256, 512],
                    channels=64,
                    n_conv_blocks=2,
                    classifier_head=classifier_head,
                    *args,
                    **kwargs)


@register_model
def convmlp_m(pretrained=False, classifier_head=True, *args, **kwargs):
    return _convmlp('convmlp_m',
                    pretrained=pretrained,
                    blocks=[3, 6, 3],
                    mlp_ratios=[3, 3, 3],
                    dims=[128, 256, 512],
                    channels=64,
                    n_conv_blocks=3,
                    classifier_head=classifier_head,
                    *args,
                    **kwargs)


@register_model
def convmlp_l(pretrained=False, classifier_head=True, *args, **kwargs):
    return _convmlp('convmlp_l',
                    pretrained=pretrained,
                    blocks=[4, 8, 3],
                    mlp_ratios=[3, 3, 3],
                    dims=[192, 384, 768],
                    channels=96,
                    n_conv_blocks=3,
                    classifier_head=classifier_head,
                    *args,
                    **kwargs)
