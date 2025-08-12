
import torch

from torch import nn

from models.layers.layer_utils import ACTIVATIONS


class MLP2d(nn.Module):

    def __init__(self, n_in, n_hid, n_out, act="relu", groups=1, kernel_size=1):
        super().__init__()

        self.conv1 = nn.Conv2d(n_in, n_hid, kernel_size, padding=kernel_size//2, groups=groups)
        self.act = ACTIVATIONS[act]()
        self.conv2 = nn.Conv2d(n_hid, n_out, kernel_size, padding=kernel_size//2, groups=groups)

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))

class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer="gelu",
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = ACTIVATIONS[act_layer]()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-3, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class conv_blockv2(nn.Module):
    def __init__(self, in_channel, out_channel, input_resolution, field_size, bias=True):
        super().__init__()
        if field_size == 1:
            self.Conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, bias=bias,),
                nn.LayerNorm([out_channel, input_resolution, input_resolution]),
            )
        elif field_size == 3:
            self.Conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, bias=bias,),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=3 // 2, bias=bias,),
                nn.LayerNorm([out_channel, input_resolution, input_resolution]),
            )
        elif field_size == 5:
            self.Conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, bias=bias,),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=3 // 2, bias=bias, ),
                nn.Conv2d(out_channel, out_channel, kernel_size=5, padding=5 // 2, bias=bias),
                nn.LayerNorm([out_channel, input_resolution, input_resolution]),
            )
        elif field_size == 7:
            self.Conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, padding = 0, bias=bias, ),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, padding= 3//2, bias=bias, ),
                nn.Conv2d(out_channel, out_channel, kernel_size=5, padding= 5//2, bias=bias),
                nn.Conv2d(out_channel, out_channel, kernel_size=7, padding= 7//2, bias=bias, ),
                nn.LayerNorm([out_channel, input_resolution, input_resolution]),
            )
        else:
            raise ValueError("The size of the input receptive field is not in the [1, 3, 5, 7] range")

    def forward(self, x):
        return self.Conv(x)

class conv_block(nn.Module):
    def __init__(self, in_channel, out_channel, input_resolution, field_size, bias=True):
        super().__init__()
        if field_size == 1:
            self.Conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, bias=bias,),
                nn.LayerNorm([out_channel, input_resolution, input_resolution]),
            )
        elif field_size == 3:
            self.Conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, bias=bias,),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=3 // 2, bias=bias,),
                nn.LayerNorm([out_channel, input_resolution, input_resolution]),
            )
        elif field_size == 5:
            self.Conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, bias=bias,),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=3 // 2, bias=bias,),
                nn.Conv2d(out_channel, out_channel, kernel_size=5, padding=5 // 2, bias=bias,),
                nn.LayerNorm([out_channel, input_resolution, input_resolution]),
            )

        elif field_size == 7:
            self.Conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, bias=bias, ),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=3 // 2, bias=bias, ),
                nn.Conv2d(out_channel, out_channel, kernel_size=5, padding=5 // 2, bias=bias, ),
                nn.Conv2d(out_channel, out_channel, kernel_size=7, padding=7 // 2, bias=bias, ),
                nn.LayerNorm([out_channel, input_resolution, input_resolution]),
            )
        else:
            raise ValueError("The size of the input receptive field is not in the [1, 3, 5] range")

    def forward(self, x):
        return self.Conv(x)



class DRF(nn.Module):
    def __init__(self, n_images, out_channels,  input_resolution, field_sizes=None, patch_sizes=None,bias=True):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.hid_features = n_images if patch_sizes is None else n_images * patch_sizes**2

        self.field_sizes = field_sizes if field_sizes is not None else [1, 3, 5]

        self.out_features = out_channels
        # self.first_conv = nn.Conv2d(self.hid_features, self.out_features, 1, bias=False)
        self.conv_blocks = nn.ModuleList(
            [
                conv_block(
                    self.hid_features, self.out_features, input_resolution, self.field_sizes[i], bias=bias
                )
                for i in range(len(self.field_sizes))
            ]
        )

    def forward(self, x,):
        f_map = []
        for blk in self.conv_blocks:
            x1 = blk(x)
            f_map.append(x1)
        x = torch.cat(f_map, dim=1)
        return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Gate(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.SiLU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_g = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc1_x = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        b, t, n, c = x.shape

        x = x.view(b*t, self.input_resolution, self.input_resolution, c)

        # 在行、列方向以 stride = 2 等间隔抽样, 实现分辨率 1/2 下采样
        x0 = x[:, 0::2, 0::2, :]  # shape = (B, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, :]  # shape = (B, H/2, W/2, C)
        x2 = x[:, 0::2, 1::2, :]  # shape = (B, H/2, W/2, C)
        x3 = x[:, 1::2, 1::2, :]  # shape = (B, H/2, W/2, C)

        # 拼接 使通道数加倍
        x = torch.cat([x0, x1, x2, x3], -1)  # shape = (B, H/2, W/2, 4*C)
        x = x.view(b*t, -1, 4 * c)  # shape = (B, H*W/4, 4*C)

        # FC 使通道数减半
        x = self.norm(x)
        x = self.reduction(x)  # shape = (B, H*W/4, 2*C)

        return x.view(b, t, -1, 2*c)

