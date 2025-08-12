import argparse
from typing import Optional, Union

import torch
import torch.nn as nn

import torch.nn.functional as F
from einops import rearrange

from models.RNN.main_frame import RNNLSTMCell
from utils.tools import str2bool


class RNN(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.configs = configs

        self.patch_size = configs.patch_size
        self.frame_channel = configs.patch_size * configs.patch_size * configs.n_image
        self.input_resolution = int(configs.input_size / configs.patch_size)
        self.num_layers = configs.num_layers
        self.num_hidden = configs.n_hidden
        self.num_patch = self.input_resolution**2

        cell_list = []

        for i in range(self.num_layers):
            in_channel = self.frame_channel if i == 0 else self.num_hidden
            cell_list.append(
                RNNLSTMCell(in_channel, self.num_hidden, configs.n_heads, self.input_resolution,
                                  configs.depth, configs.mlp_ratio, configs.drop, configs.attn_drop, configs.drop_path)
            )
        self.cell_list = nn.ModuleList(cell_list)

        self.head = nn.Sequential(
            nn.LayerNorm(self.num_hidden),
            nn.Linear(self.num_hidden, configs.n_out * self.patch_size**2,)
        )
        # shared adapter
        # self.adapter = nn.Linear(self.num_hidden, self.num_hidden, bias=False)

    @staticmethod
    def add_model_specific_args(
            parent_parser: Optional[Union[argparse.ArgumentParser, list]] = None
    ):
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]

        parser = argparse.ArgumentParser(parents=parent_parser, add_help=False)

        parser.add_argument("--context_length", type=int, default=4)
        parser.add_argument("--target_length", type=int, default=4)
        parser.add_argument("--patch_size", type=int, default=4)
        parser.add_argument("--input_size", type=int, default=32)
        parser.add_argument("--n_image", type=int, default=2)
        parser.add_argument("--n_hidden", type=int, default=256)
        parser.add_argument("--n_out", type=int, default=2)
        parser.add_argument("--n_heads", type=int, default=8)
        parser.add_argument("--num_layers", type=int, default=2)
        parser.add_argument("--fast_attn", type=str2bool, default=True)
        parser.add_argument("--mlp_ratio", type=int, default=4)
        parser.add_argument("--drop", type=float, default=0.0)
        parser.add_argument("--attn_drop", type=float, default=0.0)
        parser.add_argument("--drop_path", type=float, default=0.1)
        parser.add_argument("--depth", type=int, default=2)

        return parser

    def forward(self, frames, sampling=None):
        # [batch, length, channel, height, width]
        B, T, C, H, W = frames.shape
        frames = rearrange(frames, 'b t c (h p1) (w p2) -> b t (h w) (c p1 p2)',
                           p1=self.patch_size, p2=self.patch_size)

        next_frames = []
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []

        decouple_loss = []

        for i in range(self.num_layers):
            zeros = torch.zeros([B, self.num_patch, self.num_hidden]).to(frames.device)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        memory = torch.zeros([B, self.num_patch, self.num_hidden]).to(frames.device)

        for t in range(self.configs.context_length+self.configs.target_length - 1):

            if t < self.configs.context_length:
                if sampling and t > 0:
                    proba = (torch.rand(B) < sampling[0]).type_as(frames)[:, None, None, None]
                    net = proba * frames[:, t] + (1 - proba) * x_gen
                else:
                    net = frames[:, t]
            else:
                if sampling:
                    proba = (torch.rand(B) < sampling[1]).type_as(frames)[:, None, None, None]
                    net = proba * frames[:, t] + (1 - proba) * x_gen
                else:
                    net = x_gen

            h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](net, h_t[0], c_t[0], memory)
            delta_c_list[0] = F.normalize(delta_c, dim=1)
            delta_m_list[0] = F.normalize(delta_m, dim=1)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
                delta_c_list[i] = F.normalize(delta_c, dim=1)
                delta_m_list[i] = F.normalize(delta_m, dim=1)

            x_gen = (self.head(h_t[self.num_layers - 1].view(-1, self.num_hidden))
                     .view(B, -1, self.frame_channel))
            next_frames.append(x_gen)
            # decoupling loss
            for i in range(0, self.num_layers):
                decouple_loss.append(
                    torch.mean(torch.abs(torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=1))))

        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        # [length, batch, channel, height, width]
        next_frames = (torch.stack(next_frames, dim=0)
                       .view(T-1, B, self.input_resolution, self.input_resolution,
                             self.configs.n_out, self.patch_size, self.patch_size)
                       .permute(1, 0, 4, 2, 5, 3, 6).contiguous()
                       .view(B, T - 1, self.configs.n_out, H, W)
                       )

        return next_frames, {"decouple_loss": decouple_loss}
        # return next_frames, {}


if __name__ == "__main__":
    arg = RNN.add_model_specific_args()
    args = arg.parse_args()
    model = RNN(args)

    data = torch.rand(1, 8, 2, 32, 32)
    model.eval()
    preds, aux = model(data)
    print("of")