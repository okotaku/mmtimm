# --------------------------------------------------------
# Model from official source: https://github.com/Chenglin-Yang/LESA_classification  # noqa
# --------------------------------------------------------
import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from einops import rearrange
from mmcv.cnn import build_conv_layer
from timm.models.features import FeatureInfo
from timm.models.registry import register_model
from torch import Tensor, einsum, nn

from .helpers import mmtimm_build_model_with_cfg

default_cfgs = {
    'lesa_resnet50': {
        'url':
        'https://github.com/okotaku/mmtimm/releases/download/w_lesa/lesa_resnet50.pth'  # noqa
    },
    'lesa_wrn50': {
        'url':
        'https://github.com/okotaku/mmtimm/releases/download/w_lesa/lesa_wrn50.pth'  # noqa
    },
}


def expand_dim(t, dim, k):
    t = t.unsqueeze(dim=dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def rel_to_abs(x):
    b, h, l, _, device, dtype = *x.shape, x.device, x.dtype
    dd = {'device': device, 'dtype': dtype}
    col_pad = torch.zeros((b, h, l, 1), **dd)
    x = torch.cat((x, col_pad), dim=3)
    flat_x = rearrange(x, 'b h l c -> b h (l c)')
    flat_pad = torch.zeros((b, h, l - 1), **dd)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
    final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
    final_x = final_x[:, :, :l, (l - 1):]
    return final_x


def relative_logits_1d(q, rel_k):
    b, heads, h, w, dim = q.shape
    logits = einsum('b h x y d, r d -> b h x y r', q, rel_k)
    logits = rearrange(logits, 'b h x y r -> b (h x) y r')
    logits = rel_to_abs(logits)
    logits = logits.reshape(b, heads, h, w, w)
    logits = expand_dim(logits, dim=3, k=h)
    return logits


class LESA(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        groups,
        type,
        pe_type,
        df_channel_shrink,
        df_kernel_size,
        df_group,
        with_cp_UB_terms_only,
        kernel_size=56,
        stride=1,
        bias=False,
        dcn=None,
        **kwargs,
    ):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super().__init__()

        assert type == 'LESA'
        self.pe_type = pe_type
        self.with_cp = with_cp_UB_terms_only
        self.fmap_size = kernel_size
        self.branch_planes = out_planes

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.qk_planes = out_planes // groups // 2
        self.v_planes = self.branch_planes // groups
        kernel_size = kernel_size**2
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias

        # Multi-head self attention
        self.qkv_transform = nn.Conv1d(
            in_planes,
            (self.out_planes + self.branch_planes),
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False,
        )
        self.bn_qkv = nn.BatchNorm1d(self.out_planes + self.branch_planes)

        if pe_type == 'classification':
            self.bn_similarity = nn.BatchNorm2d(groups * 3)
            self.bn_output = nn.BatchNorm1d(self.branch_planes * 2)
        elif pe_type == 'detection_qr':
            self.bn_output = nn.BatchNorm1d(self.branch_planes)
            self.bn_similarity = nn.BatchNorm2d(groups * 2)
        else:
            raise NotImplementedError

        ReaPlanes = self.branch_planes

        if dcn is not None:
            x_layers = [
                build_conv_layer(
                    dcn,
                    in_planes,
                    self.branch_planes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    dilation=1,
                    groups=groups,
                    bias=False,
                )
            ]
        else:
            x_layers = [
                nn.Conv2d(
                    in_planes,
                    self.branch_planes,
                    kernel_size=3,
                    padding=1,
                    groups=groups,
                    bias=False,
                )
            ]

        if groups != 1:
            x_layers += [
                nn.Conv2d(
                    self.branch_planes,
                    self.branch_planes,
                    kernel_size=1,
                    bias=False,
                )
            ]
        self.x_transform = nn.Sequential(*x_layers)
        self.bn_x = nn.BatchNorm2d(self.branch_planes)

        r_layers = []
        InChannels = self.branch_planes * 2
        r_layers += [nn.ReLU(inplace=True)]
        for n_idx in range(len(df_channel_shrink)):
            r_layers += [
                nn.Conv2d(
                    InChannels,
                    int(InChannels / df_channel_shrink[n_idx]),
                    kernel_size=df_kernel_size[n_idx],
                    padding=(df_kernel_size[n_idx] - 1) // 2,
                    groups=df_group[n_idx],
                    bias=False,
                ),
                nn.BatchNorm2d(int(InChannels / df_channel_shrink[n_idx])),
                nn.ReLU(inplace=True),
            ]
            InChannels = int(InChannels / df_channel_shrink[n_idx])

        self.reasoning = nn.Sequential(*r_layers)
        TarPlanes = ReaPlanes
        proj_layers = []
        proj_layers.append(
            nn.Conv2d(
                InChannels,
                TarPlanes,
                kernel_size=df_kernel_size[-1],
                groups=df_group[-1],
                bias=False,
            ), )
        proj_layers.append(nn.BatchNorm2d(TarPlanes))
        self.projection = nn.Sequential(*proj_layers)

        # Position embedding
        if pe_type == 'classification':

            self.pe_dim = self.qk_planes * 2 + self.v_planes

            self.relative = nn.Parameter(
                torch.randn(self.pe_dim, kernel_size * 2 - 1),
                requires_grad=True,
            )

            query_index = torch.arange(kernel_size).unsqueeze(0)
            key_index = torch.arange(kernel_size).unsqueeze(1)
            relative_index = key_index - query_index + kernel_size - 1
            self.register_buffer('flatten_index', relative_index.view(-1))

        elif pe_type == 'detection_qr':

            self.pe_dim = self.qk_planes
            scale = self.pe_dim**-0.5

            self.rel_height = nn.Parameter(
                torch.randn(self.fmap_size * 2 - 1, self.pe_dim) * scale)
            self.rel_width = nn.Parameter(
                torch.randn(self.fmap_size * 2 - 1, self.pe_dim) * scale)
        else:
            raise NotImplementedError

        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def _rel_emb(self, q, rel_width, rel_height):
        h, w = self.fmap_size, self.fmap_size

        q = rearrange(q, 'b h d (x y) -> b h x y d', x=h, y=w)

        rel_logits_w = relative_logits_1d(q, rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b h x i y j-> b h (x y) (i j)')

        q = rearrange(q, 'b h x y d -> b h y x d')

        rel_logits_h = relative_logits_1d(q, rel_height)
        rel_logits_h = rearrange(rel_logits_h,
                                 'b h x i y j -> b h (y x) (j i)')
        return rel_logits_w + rel_logits_h

    def _rel_emb_ve(self, q, rel_all):
        tmp = rearrange(rel_all, 'r d -> d r').unsqueeze(0)
        tmp = expand_dim(tmp, 2, self.kernel_size)
        tmp = rel_to_abs(tmp).squeeze(0)
        return einsum('bgij, cij -> bgci', q, tmp)

    def _binary_forward(self, x):

        N, C, HW = x.shape

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N, self.groups,
                                          self.qk_planes * 2 + self.v_planes,
                                          HW),
                              [self.qk_planes, self.qk_planes, self.v_planes],
                              dim=2)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        if self.pe_type is None:
            stacked_similarity = qk
            stacked_similarity = self.bn_similarity(stacked_similarity)
        elif self.pe_type == 'detection_qr':
            stacked_similarity = qk
            qr = self._rel_emb(q, self.rel_width, self.rel_height)
            stacked_similarity = self.bn_similarity(
                torch.cat([stacked_similarity, qr],
                          dim=1)).view(N, 2, self.groups, HW, HW).sum(dim=1)
        elif self.pe_type == 'classification':
            all_embeddings = torch.index_select(
                self.relative, 1,
                self.flatten_index).view(self.qk_planes * 2 + self.v_planes,
                                         self.kernel_size, self.kernel_size)
            q_embedding, k_embedding, v_embedding = torch.split(
                all_embeddings,
                [self.qk_planes, self.qk_planes, self.v_planes],
                dim=0,
            )
            qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
            kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
            stacked_similarity = torch.cat([qk, qr, kr], dim=1)
            stacked_similarity = self.bn_similarity(stacked_similarity).view(
                N, 3, self.groups, HW, HW).sum(dim=1)
        else:
            raise NotImplementedError

        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)

        if self.pe_type == 'classification':
            sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
            stacked_binary = torch.cat([sv, sve],
                                       dim=-1).view(N, self.branch_planes * 2,
                                                    HW)
            binary = self.bn_output(stacked_binary).view(
                N, self.branch_planes, 2, HW).sum(dim=-2)
        elif self.pe_type == 'detection_qr':
            stacked_binary = sv.reshape(N, self.branch_planes, HW)
            binary = self.bn_output(stacked_binary)
        elif self.pe_type is None:
            stacked_binary = sv.reshape(N, self.branch_planes, HW)
            binary = self.bn_output(stacked_binary)
        else:
            raise NotImplementedError
        return binary

    def _unary_forward(self, x):
        unary = self.bn_x(self.x_transform(x))
        return unary

    def forward(self, x):
        # unary
        if self.with_cp:
            unary = cp.checkpoint(self._unary_forward, x)
        else:
            unary = self._unary_forward(x)

        N, C, H, W = x.shape
        x = x.view(N, C, H * W)

        # binary
        if self.with_cp:
            binary = cp.checkpoint(self._binary_forward, x)
        else:
            binary = self._binary_forward(x)

        binary = binary.view(N, self.branch_planes, H, W)

        gate_in = torch.cat([unary, binary], dim=1)
        r = self.reasoning(gate_in)
        gate = self.projection(r)
        gate = torch.sigmoid(gate)

        binary = gate * binary
        output = binary + unary

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0,
                                               math.sqrt(1. / self.in_planes))
        # nn.init.uniform_(self.relative, -0.1, 0.1)
        if self.pe_type == 'classification':
            nn.init.normal_(self.relative, 0.,
                            math.sqrt(1. / self.v_planes * 1))


def lesa3x3(**kwargs):
    return LESA(**kwargs, )


def conv3x3(in_planes: int,
            out_planes: int,
            stride: int = 1,
            groups: int = 1,
            dilation: int = 1,
            shape_kernel_size=None,
            **kwargs) -> nn.Conv2d:
    """3x3 convolution with padding."""
    # print(stride)
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        lesa=None,
        shape_kernel_size=None,
        **kwargs,
    ):
        super(Bottleneck, self).__init__()
        self.with_lesa = lesa is not None
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample
        # the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        if self.with_lesa:
            lesa = lesa if lesa is not None else {}
            self.conv2 = lesa3x3(in_planes=width,
                                 out_planes=width,
                                 kernel_size=shape_kernel_size,
                                 stride=stride,
                                 bias=False,
                                 dcn=None,
                                 **lesa,
                                 **kwargs)
        else:
            self.conv2 = conv3x3(width,
                                 width,
                                 stride,
                                 groups,
                                 dilation,
                                 shape_kernel_size=shape_kernel_size,
                                 **kwargs)
            self.bn2 = norm_layer(width)

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, token=None, **kwargs):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.with_lesa:
            out = self.conv2(out)
        else:
            out = self.conv2(out)
            out = self.bn2(out)

        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        out = self.relu(out)

        return out


class LesaResNet(nn.Module):
    def __init__(
            self,
            block,
            layers,
            lesa,
            wrn=False,
            num_classes=1000,
            zero_init_residual=False,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None,
            strides=(1, 2, 2, 1),
            stage_with_lesa=(False, False, True, True),
            stage_spatial_res=[56, 28, 14, 14],
            stage_with_first_conv=[True, True, True, False],
            features_only=False,
            out_indices=(0, 1, 2, 3),
            **kwargs,
    ):
        super().__init__()
        if wrn:
            width_per_group = width_per_group * 2

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(64)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None '
                             'or a 3-element tuple, got {}'.format(
                                 replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3,
                               self.inplanes,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        dims = [64, 128, 256, 512]
        self.block = block
        self._set_feature_info(features_only, out_indices, dims)

        self.layer1 = self._make_layer(
            block,
            dims[0],
            layers[0],
            strides[0],
            lesa=lesa if stage_with_lesa[0] else None,
            shape_kernel_size=stage_spatial_res[0],
            stage_with_first_conv=stage_with_first_conv[0],
        )

        self.layer2 = self._make_layer(
            block,
            dims[1],
            layers[1],
            stride=strides[1],
            dilate=replace_stride_with_dilation[0],
            lesa=lesa if stage_with_lesa[1] else None,
            shape_kernel_size=stage_spatial_res[1],
            stage_with_first_conv=stage_with_first_conv[1],
        )

        self.layer3 = self._make_layer(
            block,
            dims[2],
            layers[2],
            stride=strides[2],
            dilate=replace_stride_with_dilation[1],
            lesa=lesa if stage_with_lesa[2] else None,
            shape_kernel_size=stage_spatial_res[2],
            stage_with_first_conv=stage_with_first_conv[2],
        )

        self.layer4 = self._make_layer(
            block,
            dims[3],
            layers[3],
            stride=strides[3],
            dilate=replace_stride_with_dilation[2],
            lesa=lesa if stage_with_lesa[3] else None,
            shape_kernel_size=stage_spatial_res[3],
            stage_with_first_conv=stage_with_first_conv[3],
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dims[3] * block.expansion, num_classes)

        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3%
        # according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight,
                                      0)  # type: ignore[arg-type]
                else:
                    pass

    def _set_feature_info(self, features_only, out_indices, dims):
        self.features_only = features_only
        self.out_indices = out_indices
        self.feature_info_list = [
            {
                'num_chs': dims[0] * self.block.expansion,
                'reduction': 4,
                'module': None
            },
            {
                'num_chs': dims[1] * self.block.expansion,
                'reduction': 8,
                'module': None
            },
            {
                'num_chs': dims[2] * self.block.expansion,
                'reduction': 16,
                'module': None
            },
            {
                'num_chs': dims[3] * self.block.expansion,
                'reduction': 16,
                'module': None
            },
        ]
        self.feature_info = FeatureInfo(self.feature_info_list, out_indices)

    def set_features_only(self, out_indices):
        self.features_only = True
        self.out_indices = out_indices
        self.feature_info = FeatureInfo(self.feature_info_list, out_indices)

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
        dilate=False,
        lesa=None,
        shape_kernel_size=None,
        stage_with_first_conv=True,
    ):

        norm_layer = self._norm_layer
        layers = []

        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers.append(
            block(self.inplanes,
                  planes,
                  stride,
                  downsample,
                  self.groups,
                  self.base_width,
                  previous_dilation,
                  norm_layer,
                  lesa=None if stage_with_first_conv else lesa,
                  shape_kernel_size=shape_kernel_size))
        self.inplanes = planes * block.expansion

        # downsample and stride are None and 1, default values
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,
                      norm_layer=norm_layer,
                      lesa=lesa,
                      shape_kernel_size=shape_kernel_size))

        return nn.Sequential(*layers)

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_dim,
                            num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        out = []
        x = self.layer1(x)
        if self.features_only and 0 in self.out_indices:
            out.append(x)
        x = self.layer2(x)
        if self.features_only and 1 in self.out_indices:
            out.append(x)
        x = self.layer3(x)
        if self.features_only and 2 in self.out_indices:
            out.append(x)
        x = self.layer4(x)
        if self.features_only and 3 in self.out_indices:
            out.append(x)

        if self.features_only:
            return out
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        if self.features_only:
            return x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def _resnet(
    arch,
    pretrained,
    **kwargs,
):
    model = mmtimm_build_model_with_cfg(LesaResNet,
                                        arch,
                                        pretrained=pretrained,
                                        default_cfg=default_cfgs[arch],
                                        **kwargs)
    return model


@register_model
def lesa_resnet50(pretrained=False, **kwargs):
    lesa = dict(
        type='LESA',
        with_cp_UB_terms_only=False,
        pe_type='classification',  # ('classification', 'detection_qr')
        groups=8,
        df_channel_shrink=[2],  # df: dynamic fusion
        df_kernel_size=[1, 1],
        df_group=[1, 1],
    )
    model = _resnet('lesa_resnet50',
                    pretrained,
                    block=Bottleneck,
                    layers=[3, 4, 6, 3],
                    lesa=lesa,
                    **kwargs)
    return model


@register_model
def lesa_wrn50(pretrained=False, **kwargs):
    lesa = dict(
        type='LESA',
        with_cp_UB_terms_only=False,
        pe_type='classification',  # ('classification', 'detection_qr')
        groups=8,
        df_channel_shrink=[2],  # df: dynamic fusion
        df_kernel_size=[1, 1],
        df_group=[1, 1],
    )
    model = _resnet('lesa_wrn50',
                    pretrained,
                    block=Bottleneck,
                    layers=[3, 4, 6, 3],
                    lesa=lesa,
                    wrn=True,
                    **kwargs)
    return model
