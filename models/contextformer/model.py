import argparse
from typing import Optional, Union
import torch
import torch.nn as nn

from models.public import LayerScale, Mlp, Attention, PVT_embed
from utils.tools import str2bool, inverse_permutation, get_sinusoid_encoding_table


class Block(nn.Module):
    def __init__(
        self,
        dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_norm=False,
        drop=0.0, attn_drop=0.0, init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_norm=qk_norm, attn_drop=attn_drop, proj_drop=drop, norm_layer=norm_layer,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer, drop=drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class ContextFormer(nn.Module):

    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        if self.hparams.pvt:
            self.embed_images = PVT_embed(
                in_channels=self.hparams.n_image,
                out_channels=self.hparams.n_hidden,
                pretrained=self.hparams.pretrained,
                frozen=self.hparams.pvt_frozen,
            )
        else:
            self.embed_images = Mlp(
                in_features=self.hparams.n_image
                * self.hparams.patch_size
                * self.hparams.patch_size,
                hidden_features=self.hparams.n_hidden,
                out_features=self.hparams.n_hidden,
            )

        self.embed_weather = Mlp(
            in_features=self.hparams.n_weather,
            hidden_features=self.hparams.n_hidden,
            out_features=self.hparams.n_hidden,
        )

        self.mask_token = nn.Parameter(torch.zeros(self.hparams.n_hidden))

        self.blocks = nn.ModuleList(
            [
                Block(
                    self.hparams.n_hidden,
                    self.hparams.n_heads,
                    self.hparams.mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(self.hparams.depth)
            ]
        )

        self.head = Mlp(
            in_features=self.hparams.n_hidden,
            hidden_features=self.hparams.n_hidden,
            out_features=self.hparams.n_out
            * self.hparams.patch_size
            * self.hparams.patch_size,
        )

        if self.hparams.predict_delta_avg or self.hparams.predict_delta_max:
            self.head_avg = Mlp(
                in_features=self.hparams.n_hidden,
                hidden_features=self.hparams.n_hidden,
                out_features=self.hparams.n_out
                * self.hparams.patch_size
                * self.hparams.patch_size,
            )

    @staticmethod
    def add_model_specific_args(
        parent_parser: Optional[Union[argparse.ArgumentParser, list]] = None
    ):
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]

        parser = argparse.ArgumentParser(parents=parent_parser, add_help=False)

        parser.add_argument("--context_length", type=int, default=10)
        parser.add_argument("--target_length", type=int, default=20)
        parser.add_argument("--patch_size", type=int, default=4)
        parser.add_argument("--n_image", type=int, default=8)
        parser.add_argument("--n_weather", type=int, default=24)
        parser.add_argument("--n_hidden", type=int, default=256)
        parser.add_argument("--n_out", type=int, default=1)
        parser.add_argument("--n_heads", type=int, default=8)
        parser.add_argument("--depth", type=int, default=3)
        parser.add_argument("--mlp_ratio", type=float, default=4.0)
        parser.add_argument("--mtm", type=str2bool, default=False)
        parser.add_argument("--leave_n_first", type=int, default=3)
        parser.add_argument("--p_mtm", type=float, default=0.3)
        parser.add_argument("--p_use_mtm", type=float, default=0.7)
        parser.add_argument("--mask_clouds", type=str2bool, default=True)
        parser.add_argument("--use_weather", type=str2bool, default=True)
        parser.add_argument("--predict_delta", type=str2bool, default=False)
        parser.add_argument("--predict_delta0", type=str2bool, default=False)
        parser.add_argument("--predict_delta_avg", type=str2bool, default=False)
        parser.add_argument("--predict_delta_max", type=str2bool, default=False)
        parser.add_argument("--pvt", type=str2bool, default=True)
        parser.add_argument("--pretrained", type=str2bool, default=True)
        parser.add_argument("--pvt_frozen", type=str2bool, default=False)
        parser.add_argument("--add_last_ndvi", type=str2bool, default=True)
        parser.add_argument("--add_mean_ndvi", type=str2bool, default=False)
        parser.add_argument("--spatial_shuffle", type=str2bool, default=False)

        return parser

    def forward(self, data, pred_start: int = 0, preds_length: Optional[int] = None):

        # Input handling
        # preds_length=0 in the training phase and preds_length=20 in the test phase.
        preds_length = 0 if preds_length is None else preds_length

        c_l = self.hparams.context_length if self.training else pred_start  # cl=10 Use 10 time periods to predict the next 20 time periods

        hr_dynamic_inputs = data["dynamic"][0] # (B, T, C, H, W)
        hr_dynamic_mask = data["dynamic_mask"][0] # (B, T, C, H, W)

        B, T, C, H, W = hr_dynamic_inputs.shape  # Explain the specific meaning of C

        if (
            T == c_l
        ):  # If Only given Context images, add zeros (later overwritten by token mask)
            hr_dynamic_inputs = torch.cat(
                (
                    hr_dynamic_inputs,
                    torch.zeros(
                        B, preds_length, C, H, W, device=hr_dynamic_inputs.device
                    ),
                ),
                dim=1,
            )
            hr_dynamic_mask = torch.cat(
                (
                    hr_dynamic_mask,
                    torch.zeros(
                        B, preds_length, 1, H, W, device=hr_dynamic_mask.device
                    ),
                ),
                dim=1,
            )
            B, T, C, H, W = hr_dynamic_inputs.shape

        static_inputs = data["static"][0][:, :3, ...] # (B, C, H, W)

        if self.hparams.spatial_shuffle:
            perm = torch.randperm(B * H * W, device=hr_dynamic_inputs.device)
            invperm = inverse_permutation(perm)

            hr_dynamic_inputs = (
                hr_dynamic_inputs.permute(1, 2, 0, 3, 4)
                .reshape(T, C, B * H * W)[:, :, perm]
                .reshape(T, C, B, H, W)
                .permute(2, 0, 1, 3, 4)
            )

            static_inputs = (
                static_inputs.permute(1, 0, 2, 3)
                .reshape(3, B * H * W)[:, perm]
                .reshape(3, B, H, W)
                .permute(1, 0, 2, 3)
            )

        weather = data["dynamic"][1]
        if len(weather.shape) == 3:
            _, t_m, c_m = weather.shape  # Ask the c_m value
        else:
            _, t_m, c_m, h2, w2 = weather.shape
            weather = weather.reshape(B, t_m // 5, 5, c_m, h2, w2).mean(dim=2).mean(dim=(3, 4))
            _, t_m, c_m = weather.shape

        # B, C, H, W = static_inputs.shape --> B, T, C, H, W
        static_inputs = static_inputs.unsqueeze(1).repeat(1, T, 1, 1, 1)

        images = torch.cat([hr_dynamic_inputs, static_inputs], dim=2)
        B, T, C, H, W = images.shape  # c=8 5+3=8

        # Patchify
        if self.hparams.pvt:
            image_patches_embed = self.embed_images(images)
            B_patch, N_patch, C_patch = image_patches_embed.shape
        else:
            image_patches = (
                images.reshape(
                    B,
                    T,
                    C,
                    H // self.hparams.patch_size,
                    self.hparams.patch_size,
                    W // self.hparams.patch_size,
                    self.hparams.patch_size,
                )
                .permute(0, 3, 5, 1, 2, 4, 6)
                .reshape(
                    B * H // self.hparams.patch_size * W // self.hparams.patch_size,
                    T,
                    C * self.hparams.patch_size * self.hparams.patch_size,
                )
            )
            B_patch, N_patch, C_patch = image_patches.shape
            image_patches_embed = self.embed_images(image_patches)


        mask_patches = (
            hr_dynamic_mask.reshape(
                B,
                T,
                1,
                H // self.hparams.patch_size,
                self.hparams.patch_size,
                W // self.hparams.patch_size,
                self.hparams.patch_size,
            )
            .permute(0, 3, 5, 1, 2, 4, 6)
            # B, H // self.hparams.patch_size, W // self.hparams.patch_size, T,
            # 1, self.hparams.patch_size, self.hparams.patch_size,
            .reshape(
                B * H // self.hparams.patch_size * W // self.hparams.patch_size,
                T,
                1 * self.hparams.patch_size * self.hparams.patch_size,
            )
        )

        weather_patches = (
            weather.reshape(B, 1, t_m, c_m)
            .repeat(
                1, H // self.hparams.patch_size * W // self.hparams.patch_size, 1, 1
            )
            .reshape(B_patch, t_m, c_m)
        )

        # Embed Patches

        weather_patches_embed = self.embed_weather(weather_patches)

        # Add Token Mask

        if (
            self.hparams.mtm
            and self.training
            and ((torch.rand(1) <= self.hparams.p_use_mtm).all())
        ):
            token_mask = (
                (
                    torch.rand(B_patch, N_patch, device=weather_patches.device)
                    < self.hparams.p_mtm
                )
                .type_as(weather_patches) # A common way to ensure tensor type consistency and device consistency
                .reshape(B_patch, N_patch, 1)
                .repeat(1, 1, self.hparams.n_hidden)
            )
            token_mask[:, : self.hparams.leave_n_first] = 0

        else:
            token_mask = (
                torch.ones(B_patch, N_patch, device=weather_patches.device)
                .type_as(weather_patches)
                .reshape(B_patch, N_patch, 1)
                .repeat(1, 1, self.hparams.n_hidden)
            )
            token_mask[:, :c_l] = 0

        image_patches_embed[token_mask.bool()] = (
            (self.mask_token)
            .reshape(1, 1, self.hparams.n_hidden)
            .repeat(B_patch, N_patch, 1)[token_mask.bool()]
        )

        if self.hparams.mask_clouds:
            cloud_mask = (
                (mask_patches.max(-1, keepdim=True)[0] > 0)
                .bool()
                .repeat(1, 1, self.hparams.n_hidden)
            )
            image_patches_embed[cloud_mask] = (
                (self.mask_token)
                .reshape(1, 1, self.hparams.n_hidden)
                .repeat(B_patch, N_patch, 1)[cloud_mask]
            )

        # Add Image and Weather Embeddings
        if self.hparams.use_weather:
            patches_embed = image_patches_embed + weather_patches_embed
        else:
            patches_embed = image_patches_embed

        # Add Positional Embedding
        pos_embed = (
            get_sinusoid_encoding_table(N_patch, self.hparams.n_hidden)
            .to(patches_embed.device)
            .unsqueeze(0)
            .repeat(B_patch, 1, 1)
        )

        x = patches_embed + pos_embed

        # Then feed all into Transformer Encoder  (考虑x都包含什么信息）
        for blk in self.blocks:
            x = blk(x)

        # Decode image patches
        x_out = self.head(x)

        # Mask Non-masked inputs
        x_out[
            ~token_mask.bool()[
                :,
                :,
                : self.hparams.n_out
                * self.hparams.patch_size
                * self.hparams.patch_size,
            ]
        ] = -1

        # unpatchify images
        images_out = (
            x_out.reshape(
                B,
                H // self.hparams.patch_size,
                W // self.hparams.patch_size,
                N_patch,
                self.hparams.n_out,
                self.hparams.patch_size,
                self.hparams.patch_size,
            )
            .permute(0, 3, 4, 1, 5, 2, 6)
            .reshape(B, N_patch, self.hparams.n_out, H, W)
        )

        if self.hparams.add_last_ndvi:
            mask = hr_dynamic_mask[:, :c_l, ...]

            indxs = (
                torch.arange(c_l, device=mask.device)
                .expand(B, self.hparams.n_out, H, W, -1)
                .permute(0, 4, 1, 2, 3)  # B, cl, n_out, H, W
            )

            ndvi = hr_dynamic_inputs[:, :c_l, : self.hparams.n_out, ...]

            last_pixel = torch.gather(
                ndvi, 1, (indxs * (mask < 1)).argmax(1, keepdim=True)
            )  # (B, 1, n_out, H, W)

            images_out += last_pixel.repeat(1, N_patch, 1, 1, 1)

        elif self.hparams.add_mean_ndvi:
            mask = hr_dynamic_mask[:, :c_l, ...]
            ndvi = hr_dynamic_inputs[:, :c_l, : self.hparams.n_out, ...]

            mean_ndvi = (
                (ndvi * (mask < 1)).sum(1, keepdim=True)
                / ((mask < 1).sum(1, keepdim=True) + 1e-8)
            ).clamp(min=-1.0, max=1.0)

            images_out += mean_ndvi.repeat(1, N_patch, 1, 1, 1)

        if self.hparams.predict_delta_avg:

            image_avg = self.head_avg(x[:, :c_l, :].mean(1).unsqueeze(1))
            image_avg_out = (
                image_avg.reshape(
                    B,
                    H // self.hparams.patch_size,
                    W // self.hparams.patch_size,
                    1,
                    self.hparams.n_out,
                    self.hparams.patch_size,
                    self.hparams.patch_size,
                )
                .permute(0, 3, 4, 1, 5, 2, 6)
                .reshape(B, 1, self.hparams.n_out, H, W)
            )

            images_out += image_avg_out.repeat(1, N_patch, 1, 1, 1)

        elif self.hparams.predict_delta_max:
            image_avg = self.head_avg(x[:, :c_l, :].max(1)[0]).unsqueeze(1)
            image_avg_out = (
                image_avg.reshape(
                    B,
                    H // self.hparams.patch_size,
                    W // self.hparams.patch_size,
                    1,
                    self.hparams.n_out,
                    self.hparams.patch_size,
                    self.hparams.patch_size,
                )
                .permute(0, 3, 4, 1, 5, 2, 6)
                .reshape(B, 1, self.hparams.n_out, H, W)
            )

            images_out += image_avg_out.repeat(1, N_patch, 1, 1, 1)

        elif self.hparams.predict_delta:
            images_out[:, 0, ...] += images[:, 0, : self.hparams.n_out, ...]
            images_out = torch.cumsum(images_out, 1)
        elif self.hparams.predict_delta0:
            images_out += (images[:, :1, : self.hparams.n_out, ...]).repeat(
                1, N_patch, 1, 1, 1
            )

        if not self.training:
            images_out = images_out[:, -preds_length:]

        if self.hparams.spatial_shuffle:
            B, T, C, H, W = images_out.shape
            images_out = (
                images_out.permute(1, 2, 0, 3, 4)
                .reshape(T, C, B * H * W)[:, :, invperm]
                .reshape(T, C, B, H, W)
                .permute(2, 0, 1, 3, 4)
            )
        images_out = images_out[:, -self.hparams.target_length:]
        return images_out, {}


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    arg = ContextFormer.add_model_specific_args()
    args = arg.parse_args()
    model = ContextFormer(args)
    model.to(device=device)
    data = {
        "dynamic": [
            torch.rand(1, 30, 5, 64, 64),
            torch.rand(1, 30, 24),
        ],
        "dynamic_mask": [torch.rand(1, 30, 1, 64, 64)],
        "static": [torch.rand(1, 3, 64, 64)],
    }
    for key, value in data.items():
        if isinstance(value, list):
            for i in range(len(value)):
                if isinstance(value[i], torch.Tensor):
                    data[key][i] = value[i].to(device)
        elif isinstance(value, torch.Tensor):
            data[key] = value.to(device)
    model.eval()
    preds, aux = model(data)
    print("of")
