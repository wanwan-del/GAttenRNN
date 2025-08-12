import argparse
import ast
import warnings
from typing import Optional, Union, Final
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F
from models.DynamicNet.main_frame import Early_feature, Mid_feature
from models.layers.blocks import Mlp
from utils.tools import get_sinusoid_encoding_table, str2bool

warnings.filterwarnings("ignore", category=UserWarning)



class DynamicNet_multi(nn.Module):

    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.input_resolution = int(self.hparams.input_size / self.hparams.patch_size)
        self.num_patch = self.input_resolution ** 2
        self.total_T = self.hparams.context_length + self.hparams.target_length

        self.early_feature = Early_feature(
            in_channels=self.hparams.n_image, out_channels=self.hparams.n_hidden,
            input_resolution=self.input_resolution, act=self.hparams.re_act,
            patch_size=self.hparams.patch_size, field_size=self.hparams.field_size,
        )

        self.embed_weather = Mlp(
            in_features=self.hparams.n_weather, hidden_features=self.hparams.n_hidden,
            out_features=self.hparams.n_hidden,
        )
        self.mask_token = nn.Parameter(torch.zeros(self.hparams.n_hidden))

        self.pos_embed = nn.Parameter(
            get_sinusoid_encoding_table(self.total_T * self.num_patch, self.hparams.n_hidden, T=10000),
            requires_grad=False,
        ).view(1, self.total_T, self.num_patch, self.hparams.n_hidden)

        self.mid_feature = nn.ModuleList(
            [
                Mid_feature(dim=self.hparams.n_hidden, num_heads=self.hparams.n_heads,
                            drop=self.hparams.drop, attn_drop=self.hparams.attn_drop,
                            drop_path=self.hparams.drop_path,
                            )
                for _ in range(self.hparams.depth)
            ]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(self.hparams.n_hidden,),
            nn.Linear(self.hparams.n_hidden, self.hparams.n_out * self.hparams.patch_size * self.hparams.patch_size, )
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
        parser.add_argument("--n_image", type=int, default=8)
        parser.add_argument("--n_hidden", type=int, default=256)
        parser.add_argument("--patch_size", type=int, default=8)
        parser.add_argument("--input_size", type=int, default=128)
        parser.add_argument("--n_out", type=int, default=1)
        parser.add_argument("--add_last_ndvi", type=str2bool, default=True)

        parser.add_argument("--mtm", type=str2bool, default=True)
        parser.add_argument("--leave_n_first", type=int, default=3)
        parser.add_argument("--p_mtm", type=float, default=0.7)
        parser.add_argument("--p_use_mtm", type=float, default=0.5)

        parser.add_argument("--n_weather", type=int, default=24)

        parser.add_argument("--re_act", type=str, default="silu")
        parser.add_argument("--field_size", type=ast.literal_eval, default=[3, 5, 7])
        parser.add_argument("--mlp_ratio", type=int, default=4)
        parser.add_argument("--n_heads", type=int, default=8)
        parser.add_argument("--depth", type=int, default=3)
        parser.add_argument("--fast_attn", type=str2bool, default=True)
        parser.add_argument("--drop", type=float, default=0.0)
        parser.add_argument("--attn_drop", type=float, default=0.0)
        parser.add_argument("--drop_path", type=float, default=0.0)


        return parser

    def forward(self, data):

        hr_dynamic_inputs = data["dynamic"][0]  # (B, T, C, H, W)
        hr_dynamic_mask = data["dynamic_mask"][0]  # (B, T, C, H, W)
        static_inputs = data["static"][0][:, :3, ...]  # (B, C, H, W)
        weather = data["dynamic"][1]

        B, T, C, H, W = hr_dynamic_inputs.shape  # Explain the specific meaning of C

        if T == self.hparams.context_length:
            hr_dynamic_inputs = torch.cat(
                (hr_dynamic_inputs,
                    torch.zeros(B, self.hparams.target_length, C, H, W, device=hr_dynamic_inputs.device)),
                dim=1)
            hr_dynamic_mask = torch.cat(
                (hr_dynamic_mask,
                    torch.zeros(B, self.hparams.target_length, 1, H, W, device=hr_dynamic_mask.device)),
                dim=1)

        # B, C, H, W = static_inputs.shape --> B, T, C, H, W
        static_inputs = static_inputs.unsqueeze(1).repeat(1, self.total_T, 1, 1, 1)

        images = torch.cat([hr_dynamic_inputs, static_inputs], dim=2)

        # Patchify
        image_patches_embed = self.early_feature(images)
        image_patches_embed = rearrange(image_patches_embed, 'b t n c -> (b n) t c',)
        B_patch = image_patches_embed.shape[0]
        if self.hparams.mtm and self.training and ((torch.rand(1) <= self.hparams.p_use_mtm).all()):
            token_mask = (
                (torch.rand(B_patch, self.total_T, device=image_patches_embed.device) < self.hparams.p_mtm)
                .type_as(image_patches_embed)  # A common way to ensure tensor type consistency and device consistency
                .view(B_patch, self.total_T, 1)
                .repeat(1, 1, self.hparams.n_hidden)
            )
            token_mask[:, :self.hparams.leave_n_first] = 0

        else:
            token_mask = (
                torch.ones(B_patch, self.total_T, device=image_patches_embed.device)
                .type_as(image_patches_embed)
                .view(B_patch, self.total_T, 1)
                .repeat(1, 1, self.hparams.n_hidden)
            )
            token_mask[:, :self.hparams.context_length] = 0
        image_patches_embed[token_mask.bool()] = (
            self.mask_token
            .view(1, 1, self.hparams.n_hidden)
            .repeat(B_patch, self.total_T, 1)[token_mask.bool()]
        )

        mask_patches = rearrange(hr_dynamic_mask, 'b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)',
                                 p1=self.hparams.patch_size, p2=self.hparams.patch_size )
        cloud_mask = (
            (mask_patches.max(-1, keepdim=True)[0] > 0)
            .bool()
            .repeat(1, 1, self.hparams.n_hidden)
        )
        image_patches_embed[cloud_mask] = (
            (self.mask_token)
            .reshape(1, 1, self.hparams.n_hidden)
            .repeat(B_patch, self.total_T, 1)[cloud_mask]
        )

        # Add Image and Weather Embeddings
        image_patches_embed = rearrange(image_patches_embed, '(b n) t c -> b t n c', n=self.num_patch)
        weather_patches_embed = self.embed_weather(weather).view(B, self.total_T, 1, self.hparams.n_hidden)

        patches_embed = image_patches_embed + weather_patches_embed
        x = patches_embed + self.pos_embed.to(images.device)

        for blk in self.mid_feature:
            x = blk(x)

        x_out = self.head(x.view(-1, self.hparams.n_hidden))
        x_out = rearrange(x_out, '(b t n) c -> (b n) t c', t=self.total_T, n=self.num_patch)
        x_out[~token_mask.bool()[
               :, :, :self.hparams.n_out * self.hparams.patch_size * self.hparams.patch_size,
             ]
        ] = -1

        images_out = rearrange(x_out, '(b h w) t (c p1 p2) -> b t c (h p1) (w p2)',
                               h=self.input_resolution, w=self.input_resolution,
                               p1=self.hparams.patch_size, p2=self.hparams.patch_size
                               )

        if self.hparams.add_last_ndvi:
            mask = hr_dynamic_mask[:, :self.hparams.context_length, ...]

            indxs = (
                torch.arange(self.hparams.context_length, device=mask.device)
                .expand(B, self.hparams.n_out, H, W, -1)
                .permute(0, 4, 1, 2, 3)  # B, cl, n_out, H, W
            )

            ndvi = hr_dynamic_inputs[:, :self.hparams.context_length, :self.hparams.n_out, ...]

            last_pixel = torch.gather(
                ndvi, 1, (indxs * (mask < 1)).argmax(1, keepdim=True)
            )  # (B, 1, n_out, H, W)

            images_out += last_pixel.repeat(1, self.total_T, 1, 1, 1)

        return images_out, {}

if __name__ == "__main__":
    arg = DynamicNet_multi.add_model_specific_args()
    args = arg.parse_args()
    model = DynamicNet_multi(args)
    data = {
        "dynamic": [
            torch.rand(1, 30, 5, 128, 128),
            torch.rand(1, 30, 24),
        ],
        "dynamic_mask": [torch.rand(1, 30, 1, 128, 128)],
        "static": [torch.rand(1, 3, 128, 128)],
    }
    model.eval()
    preds, aux = model(data)
    print("of")


