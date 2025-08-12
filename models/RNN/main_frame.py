
import torch
import torch.nn as nn
from timm.layers import DropPath, trunc_normal_

from models.layers.blocks import Gate, PreNorm
from models.layers.attention import Attention
from models.layers.layer_utils import get_sinusoid_encoding_table


class GatedTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_ratio, dropout=0., attn_dropout=0., drop_path=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, num_heads=heads, proj_drop=dropout, attn_drop=attn_dropout)),
                PreNorm(dim, Gate(dim, hidden_features=mlp_ratio*dim, drop=dropout)),
                DropPath(drop_path) if drop_path > 0. else nn.Identity(),
                DropPath(drop_path) if drop_path > 0. else nn.Identity()
            ]))
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
        for attn, ff, drop_path1, drop_path2 in self.layers:
            x = x + drop_path1(attn(x))
            x = x + drop_path2(ff(x))
        return self.norm(x)


class RNNLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, num_heads, input_resolution, depth=1, mlp_ratio=4, drop_out=0.0,
                 attn_drop_out=0.0, drop_path=0.1):
        super().__init__()

        self.input_resolution = input_resolution

        self.linear_xh = nn.Linear(num_hidden + in_channel, num_hidden,)
        self.linear_xm = nn.Linear(num_hidden + in_channel, num_hidden,)

        self.Atten_c = GatedTransformer(dim=num_hidden, depth=depth,  heads=num_heads,
                                        mlp_ratio=mlp_ratio,
                                        dropout=drop_out, attn_dropout=attn_drop_out,
                                        drop_path=drop_path,)

        self.Atten_m = GatedTransformer(num_hidden, depth,  num_heads, mlp_ratio, drop_out, attn_dropout=attn_drop_out,
                                          drop_path=drop_path,)

        self.pos_embed = nn.Parameter(
            get_sinusoid_encoding_table(input_resolution * input_resolution, num_hidden, T=10000),
            requires_grad=False,
        ).view(1, input_resolution*input_resolution, num_hidden)

        self.linear_o = nn.Sequential(
            nn.Linear(num_hidden * 2, num_hidden,),
            nn.Dropout(drop_out)
        )

        self.last_linear = nn.Linear(num_hidden * 2, num_hidden, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):

        xh = self.linear_xh(torch.cat((x_t, h_t), dim=-1)) + self.pos_embed.to(x_t.device)
        xm = self.linear_xm(torch.cat((x_t, m_t), dim=-1)) + self.pos_embed.to(x_t.device)

        Ft = self.Atten_c(xh)
        ft = self.Atten_m(xm)

        Gate = torch.tanh(Ft)
        gate = torch.tanh(ft)

        Cell = torch.sigmoid(Ft)
        cell = torch.sigmoid(ft)

        delta_m = gate * cell
        delta_c = Gate * Cell

        c_new = Gate * (c_t + Cell)
        m_new = gate * (m_t + cell)

        mem = torch.cat((c_new, m_new), dim=-1)

        o_t = torch.sigmoid(Ft + ft + self.linear_o(mem))
        h_new = o_t * torch.tanh(self.last_linear(mem))
        return h_new, c_new, m_new, delta_c, delta_m