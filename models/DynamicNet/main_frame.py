from timm.layers import DropPath, trunc_normal_
from torch import nn
from einops.layers.torch import Rearrange

from models.DynamicNet.swin_atten import SwinTransformerBlock
from models.layers.blocks import DRF, Gate, Mlp
from models.layers.layer_utils import ACTIVATIONS
from models.layers.attention import Attention


class Early_feature(nn.Module):
    def __init__(self, encoder,in_channels, out_channels, input_resolution, patch_size, act='silu', field_size=None, bias=True):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.input_resolution = input_resolution
        self.encoder =encoder
        self.reshape = Rearrange('b t c (h p1) (w p2) -> (b t) (p1 p2 c) h w', p1=self.patch_size, p2=self.patch_size)
        if self.encoder == "Conv2d":
            field_size = [1, 3, 5] if field_size is None else field_size
            self.local = DRF(n_images=in_channels, out_channels=out_channels, patch_sizes=self.patch_size,
                             field_sizes=field_size, input_resolution=input_resolution, bias=bias)
            self.act = ACTIVATIONS[act]()
            proj_channels = out_channels * len(field_size)

            self.project = nn.Conv2d(in_channels=proj_channels, out_channels=out_channels, kernel_size=1, )
            self.apply(self._init_weights)

        elif self.encoder == "Linear":
            self.mlp = Mlp(in_channels*self.patch_size**2 , out_channels, out_channels)



    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = self.reshape(x)

        if self.encoder == "Conv2d":
            map = self.act(self.local(x))
            x = self.project(map)

        x = x.view(B * T, -1, self.input_resolution*self.input_resolution).permute(0, 2, 1).contiguous()

        if self.encoder == "Linear":
            x = self.mlp(x)

        x_patches = x.view(B, T, self.input_resolution*self.input_resolution, self.out_channels)

        return x_patches  # (B, T, N, C)


class Mid_feature(nn.Module):
    def __init__(
        self,
        dim, input_resolution, num_heads, mlp_ratio=4, drop=0.0, attn_drop=0.0,
        norm_layer=nn.LayerNorm, fast_attn=True, use_swin=False, window_size=4,
        drop_path=0.1, i=0, Gate_act="silu", Space=True,
    ):
        super().__init__()
        if Gate_act == "silu":
            gate = Gate
        elif Gate_act == "gelu":
            gate = Mlp

        self.Tem_attn = Attention(
            dim, num_heads=num_heads,
            attn_drop=attn_drop, proj_drop=drop, norm_layer=norm_layer, fast_attn=fast_attn,
        )
        self.Tem_drop1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.Tem_gate = gate(dim, mlp_ratio * dim, drop=drop)
        self.Tem_drop2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.Tem_norm1 = norm_layer(dim)
        self.Tem_norm2 = norm_layer(dim)
        self.Tem_norm3 = norm_layer(dim)

        self.Space = Space
        if self.Space:
            if use_swin:
                shift_size = 0 if (i % 2 == 0) else window_size // 2
                self.Spa_attn = SwinTransformerBlock(
                    dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size,
                    shift_size=shift_size, drop=drop, attn_drop=attn_drop,
                )
            else:
                self.Spa_attn = Attention(
                    dim, num_heads=num_heads,
                    attn_drop=attn_drop, proj_drop=drop, norm_layer=norm_layer, fast_attn=fast_attn,
                )

            self.Spa_drop1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.Spa_gate = gate(dim, mlp_ratio * dim, drop=drop)
            self.Spa_drop2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.Spa_norm1 = norm_layer(dim)
            self.Spa_norm2 = norm_layer(dim)
            self.Spa_norm3 = norm_layer(dim)


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        b, t, n, c = x.shape

        x = x.transpose(1, 2).reshape(-1, t, c)
        x = x + self.Tem_drop1(self.Tem_attn(self.Tem_norm1(x)))
        x = x + self.Tem_drop2(self.Tem_gate(self.Tem_norm2(x)))
        x = self.Tem_norm3(x)

        x = x.view(b, n, t, c).transpose(1, 2).reshape(-1, n, c)

        if self.Space:
            x = x + self.Spa_drop1(self.Spa_attn(self.Spa_norm1(x)))
            x = x + self.Spa_drop2(self.Spa_gate(self.Spa_norm2(x)))
            x = self.Spa_norm3(x)

        return x.view(b, t, n, c)

