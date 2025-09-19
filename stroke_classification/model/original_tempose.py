# TemPose: a new skeleton-based transformer model designed for fine-grained motion recognition in badminton
# (2023/08) https://ieeexplore.ieee.org/document/10208321
# Authors: Magnus Ibh, Stella Grasshof, Dan Witzner, Pascal Madeleine

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange, repeat

import numpy as np

from torchinfo import summary


def get_2d_sincos_pos_embed(embed_dim, seq_len, cls_token=False):
    "Used from MAE paper"
    pos = np.arange(seq_len, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    "Used from MAE paper"
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)  # modified
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D) ## specific pose connection possible here.
    return emb


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, mask=None, y=None):
        return self.fn(self.norm(x), mask)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=48, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        # q, k, v: (b*n, h, t, d_head)

        dots: Tensor = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # dots: (b*n, h, t, t)
        if mask is not None:
            dots = dots.masked_fill(mask == 0, -1e9) 

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        # out: (b*n, h, t, d_head)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # out: (b*n, t, d_head*n_head)
        return self.to_out(out)  # (b*n, t, d)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask) + x
            x = ff(x) + x
        return x


class TemPoseII_TF(nn.Module):
    def __init__(self, *,
        poses_numbers, time_steps, num_people, num_classes,
        dim=50, kernel_size=5, depth=4, depth_int=3, heads=6,
        scale_dim=4, dim_head=75, dropout=0.3, emb_dropout=0.3
    ):
        super().__init__()

        self.pool = 'cls'
        self.heads = heads
        self.time_sequence = time_steps
        self.people = num_people + 3
        self.to_patch_embedding = nn.Linear(poses_numbers, dim)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_embedding = nn.Parameter(torch.zeros(1, self.people, time_steps+1, dim), requires_grad=True)
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        self.interaction_token = nn.Parameter(torch.randn(1, 1, dim))
        self.interaction_embedding = nn.Parameter(torch.zeros(1, self.people+1, dim))
        self.interaction_transformer = Transformer(dim, depth_int, heads, dim_head, dim * scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, scale_dim * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(scale_dim * dim, num_classes)
        )

        ### TCN block
        self.num_channels = [dim // 2, dim]
        self.kernel_size = kernel_size
        input_size = 2

        # define temporal convolutional layers
        layers = []
        num_levels = len(self.num_channels)
        for i in range(num_levels):
            dilation_size = (2 * i) + 1
            in_channels = input_size if i == 0 else self.num_channels[i-1]
            out_channels = self.num_channels[i]
            padding = (kernel_size - 1) * dilation_size // 2
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout)
            ]
        self.tcn1 = nn.Sequential(*layers)
        self.tcn2 = nn.Sequential(*layers)
        self.tcn3 = nn.Sequential(*layers)
        
        ### weight initialization
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        temp_embed = repeat(torch.from_numpy(get_2d_sincos_pos_embed(self.temporal_embedding.shape[-1], int(self.time_sequence), cls_token=True)).float().unsqueeze(0),'() t d -> n t d', n=self.people)
        self.temporal_embedding.data.copy_(temp_embed.unsqueeze(0))

        int_embed = get_2d_sincos_pos_embed(self.interaction_embedding.shape[-1], self.people, cls_token=True)
        self.interaction_embedding.data.copy_(torch.from_numpy(int_embed).float().unsqueeze(0))
        
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.temporal_token, std=.02)
        torch.nn.init.normal_(self.interaction_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x, sp, t_pad=None):  # ,t_pad,n_pad
        x = self.to_patch_embedding(x)
        # x: (b, n, t, dim)
        b, n, t, _ = x.shape  ### t is num frames # n is number of people
        n = self.people

        input_seq = rearrange(sp,'b t d -> b d t')
        x_pos1 = rearrange(self.tcn1(input_seq[:, :2, :]), 'b d t -> b t d').unsqueeze(1)
        x_pos2 = rearrange(self.tcn2(input_seq[:, 2:4, :]), 'b d t -> b t d').unsqueeze(1)
        x_shuttle = rearrange(self.tcn3(input_seq[:,-2:,:]), 'b d t -> b t d').unsqueeze(1)
        x = torch.cat((x, x_pos1, x_pos2, x_shuttle), dim=1)
        # x: (b, n, t, dim), n=self.people

        cls_temporal_tokens = repeat(self.temporal_token, '() t d -> b n t d', b=b, n=n)
        x = torch.cat((cls_temporal_tokens, x), dim=2)
        # x: (b, n, 1+t, dim)
        x += self.temporal_embedding[:, :, :(t + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b n t d -> (b n) t d')
        mask_t = torch.ones((b * n, t+1)).type(torch.LongTensor)
        t_pad = repeat(t_pad,'(t ) -> (t d)', d=n).type(torch.LongTensor)  # I think 't' here is not time steps but the size of the batch.
        for j, index in enumerate(t_pad):
            mask_t[j, (index+1):] = 0  # so that makes sense for the mask setting the values in time steps after the cls token and real data to 0.
        mask_t = mask_t.to(x.device)
        x = self.temporal_transformer(x, mask_t.unsqueeze(1).unsqueeze(1))
        # x: (b*n, 1+t, dim)
        x = rearrange(x[:, 0], '(b n) ... -> b n ...', b=b)
        # x: (b, n, dim)

        cls_interaction_tokens = repeat(self.interaction_token, '() t d -> b t d', b=b)
        x = torch.cat((cls_interaction_tokens, x), dim=1)
        # x: (b, 1+n, dim)
        x += self.interaction_embedding[:, :(n+1)]
        x = self.interaction_transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        # x: (b, dim)
        return self.mlp_head(x)

    def predict(self, x, sp, pad=None):
        # Apply softmax to output. 
        pred = F.softmax(self.forward(x, sp, pad), dim=1).max(1)[1].cpu()


# helpers
if __name__ == '__main__':
    b, t, n = 3, 30, 2
    n_features = (17 + 19 * 1) * n
    pose = torch.randn((b, t, n, n_features), dtype=torch.float)
    pos = torch.randn((b, t, n, 2), dtype=torch.float)
    shuttle = torch.randn((b, t, 2), dtype=torch.float)
    t_pad = torch.tensor([t], dtype=torch.long).repeat(b)

    sp = torch.concat((pos.flatten(start_dim=2), shuttle), dim=-1)

    input_data = [
        pose.transpose(1, 2).contiguous(),
        sp,
        t_pad
    ]
    model = TemPoseII_TF(
        poses_numbers=n_features,
        time_steps=30,
        num_people=2,
        num_classes=35,
        dim=100,
        depth=2,
        depth_int=2,
        dim_head=128,
        emb_dropout=0.3
    )
    summary(model, input_data=input_data, depth=4, device='cpu')
