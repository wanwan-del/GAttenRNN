import argparse
import ast
import warnings
from typing import Optional, Union
import torch
import torch.nn as nn

from models.DynamicNet.main_frame import Early_feature, Mid_feature
from models.layers.blocks import Mlp
from utils.tools import str2bool, get_sinusoid_encoding_table

warnings.filterwarnings("ignore", category=UserWarning)


class DynamicNet_sigle(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.input_resolution = int(self.hparams.input_size / self.hparams.patch_size)

        self.num_patch = self.input_resolution**2

        self.early_feature = Early_feature(
            in_channels=self.hparams.n_image, out_channels=self.hparams.n_hidden,
            input_resolution=self.input_resolution, act=self.hparams.act,
            patch_size=self.hparams.patch_size, field_size=self.hparams.field_size,
            encoder=self.hparams.encoder
        )

        self.mask_token = nn.Parameter(torch.zeros(self.hparams.n_hidden))
        self.total_T = self.hparams.context_length + self.hparams.target_length

        if self.hparams.Space:
            if self.hparams.use_swin:
                self.pos_embed = nn.Parameter(
                    get_sinusoid_encoding_table(self.total_T, self.hparams.n_hidden, T=1000),
                    requires_grad=False,
                ).view(1, self.total_T, 1, self.hparams.n_hidden)
            else:
                self.pos_embed = nn.Parameter(
                    get_sinusoid_encoding_table(self.total_T * self.num_patch, self.hparams.n_hidden, T=10000),
                    requires_grad=False,
                ).view(1, self.total_T, self.num_patch, self.hparams.n_hidden)
        else:
            self.pos_embed = nn.Parameter(
                get_sinusoid_encoding_table(self.total_T, self.hparams.n_hidden, T=1000),
                requires_grad=False,
            ).view(1, self.total_T, 1, self.hparams.n_hidden)



        self.mid_feature = nn.ModuleList(
                [
                    Mid_feature(dim=self.hparams.n_hidden, num_heads=self.hparams.n_heads,
                                drop=self.hparams.drop, attn_drop=self.hparams.attn_drop,
                                use_swin=self.hparams.use_swin, window_size=self.hparams.window_size,
                                drop_path=self.hparams.drop_path, input_resolution=self.input_resolution,
                                i=i, Space=self.hparams.Space, Gate_act=self.hparams.Gate_act)
                    for i in range(self.hparams.depth)
                ]
            )
        if self.hparams.mlp:
            self.head = Mlp(
                in_features=self.hparams.n_hidden,
                hidden_features=self.hparams.n_hidden,
                out_features=self.hparams.n_out
                             * self.hparams.patch_size
                             * self.hparams.patch_size,
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(self.hparams.n_hidden, ),
                nn.Linear(self.hparams.n_hidden,
                          self.hparams.n_out * self.hparams.patch_size * self.hparams.patch_size, )
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
        parser.add_argument("--target_length", type=int, default=10)
        parser.add_argument("--patch_size", type=int, default=8)
        parser.add_argument("--input_size", type=int, default=128)
        parser.add_argument("--n_image", type=int, default=1)
        parser.add_argument("--n_hidden", type=int, default=256)
        parser.add_argument("--n_out", type=int, default=1)
        parser.add_argument("--n_heads", type=int, default=8)
        parser.add_argument("--depth", type=int, default=4)
        parser.add_argument("--act", type=str, default="silu")
        parser.add_argument("--fast_attn", type=str2bool, default=True)
        parser.add_argument("--mlp_ratio", type=int, default=4)
        parser.add_argument("--drop", type=float, default=0.1)
        parser.add_argument("--attn_drop", type=float, default=0.1)
        parser.add_argument("--drop_path", type=float, default=0.1)
        parser.add_argument("--mtm", type=str2bool, default=False)
        parser.add_argument("--p_use_mtm", type=float, default=0.5)
        parser.add_argument("--p_mtm", type=float, default=0.7)
        parser.add_argument("--leave_n_first", type=int, default=3)
        parser.add_argument("--field_size", type=ast.literal_eval, default=[1, 3, 5])
        parser.add_argument("--encoder_bias", type=str2bool, default=True)
        parser.add_argument("--use_swin", type=str2bool, default=False)
        parser.add_argument("--window_size", type=int, default=4)

        ## For ablation experiments
        parser.add_argument("--mlp", type=str2bool, default=False)
        parser.add_argument("--encoder", type=str, default="Conv2d")
        parser.add_argument("--Gate_act", type=str, default="silu")
        parser.add_argument("--Space", type=str2bool, default=True)


        return parser

    def forward(self, data):
        B, T, C, H, W = data.shape  # Explain the specific meaning of C

        if T == self.hparams.context_length and self.hparams.target_length > 0:
            data = torch.cat(
                (data,torch.zeros(B, self.hparams.target_length, C, H, W, device=data.device)),
                dim=1)

        patches_embed = self.early_feature(data)

        if self.hparams.mtm and self.training and ((torch.rand(1) <= self.hparams.p_use_mtm).all()):
            token_mask = (
                (
                    torch.rand(B, self.total_T, self.num_patch, device=patches_embed.device)
                    < self.hparams.p_mtm
                )
                .type_as(patches_embed) # A common way to ensure tensor type consistency and device consistency
                .reshape(B, self.total_T, self.num_patch, 1)
                .repeat(1, 1, 1, self.hparams.n_hidden)
            )
            token_mask[:, :self.hparams.leave_n_first] = 0

        else:
            token_mask = (
                torch.ones(B, self.total_T, self.num_patch, device=patches_embed.device)
                .type_as(patches_embed)
                .reshape(B, self.total_T, self.num_patch, 1)
                .repeat(1, 1, 1, self.hparams.n_hidden)
            )
            token_mask[:, :self.hparams.context_length] = 0

        patches_embed[token_mask.bool()] = (
            (self.mask_token)
            .reshape(1, 1, 1, self.hparams.n_hidden)
            .repeat(B, self.total_T, self.num_patch, 1)[token_mask.bool()]
        )

        x = patches_embed + self.pos_embed.to(data.device)

        for blk in self.mid_feature:
            x = blk(x)

        x_out = self.head(x.view(B, -1, self.hparams.n_hidden))
        x_out = x_out.view(B, self.total_T, -1, self.hparams.n_out * self.hparams.patch_size * self.hparams.patch_size)
        # Mask Non-masked inputs
        x_out[ ~token_mask.bool()[:, :, :,
             :self.hparams.n_out * self.hparams.patch_size * self.hparams.patch_size,
             ]
        ] = -1

        images_out = (
            x_out.reshape(
                B, self.total_T, self.input_resolution, self.input_resolution, self.hparams.n_out, self.hparams.patch_size, self.hparams.patch_size,
            )
            .permute(0, 1, 4, 2, 5, 3, 6)
            .reshape(B, self.total_T, self.hparams.n_out, H, W)
        )

        return images_out, {}


if __name__ == "__main__":
    arg = DynamicNet_sigle.add_model_specific_args()
    args = arg.parse_args()
    model = DynamicNet_sigle(args)
    data = torch.rand(1, 20, 1, 128, 128)
    model.eval()
    preds, aux = model(data)
    print("of")


