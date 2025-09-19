# SkateFormer: Skeletal-Temporal Transformer for Human Action Recognition
# (2024/07, v3) https://arxiv.org/abs/2403.09508
# Authors: Jeonghyeok Do, Munchurl Kim

# Modified by Jing-Yuan Chang

import torch
from torch import Tensor, nn
from torchinfo import summary

from timm.models.layers import Mlp, DropPath

import math


def type_1_partition(x: Tensor, partition_size: tuple[int, int]):
    '''(B, C, T, V) => (B * M * K, N, L, C)

    Parameters
    - partition_size = (N frames, L joints)
        - There are M segments and K joint-groups.
        - T = M * N
        - V = K * L
    '''
    B, C, T, V = x.shape
    N, L = partition_size
    partitions = x.view(B, C, T // N, N, V // L, L)
    partitions = partitions.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, N, L, C)
    return partitions  # (B*M*K, N, L, C)


def type_1_reverse(
    partitions: Tensor,
    original_size: tuple[int, int],
    partition_size: tuple[int, int]
):
    '''(B * M * K, N, L, C) => (B, C, T, V)

    Parameters
    - original_size = (T, V)
    - partition_size = (N frames, L joints)
        - There are M segments and K joint-groups.
        - T = M * N
        - V = K * L
    '''
    T, V = original_size
    N, L = partition_size
    M, K = T // N, V // L
    B = partitions.shape[0] // (M * K)
    output = partitions.view(B, M, K, N, L, -1)
    output = output.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, T, V)
    return output  # (B, C, T, V)


def type_2_partition(x: Tensor, partition_size: tuple[int, int]):
    '''(B, C, T, V) => (B * M * L, N, K, C)

    Parameters
    - partition_size = (N frames, K distant joints)
        - There are M segments and L distant joint-groups.
        - T = M * N
        - V = K * L
    '''
    B, C, T, V = x.shape
    N, K = partition_size
    partitions = x.view(B, C, T // N, N, K, V // K)
    partitions = partitions.permute(0, 2, 5, 3, 4, 1).contiguous().view(-1, N, K, C)
    return partitions  # (B*M*L, N, K, C)


def type_2_reverse(
    partitions: Tensor,
    original_size: tuple[int, int],
    partition_size: tuple[int, int]
):
    '''(B * M * L, N, K, C) => (B, C, T, V)

    Parameters
    - original_size = (T, V)
    - partition_size = (N frames, K distant joints)
        - There are M segments and L distant joint-groups.
        - T = M * N
        - V = K * L
    '''
    T, V = original_size
    N, K = partition_size
    M, L = T // N, V // K
    B = partitions.shape[0] // (M * L)
    output = partitions.view(B, M, L, N, K, -1)
    output = output.permute(0, 5, 1, 3, 4, 2).contiguous().view(B, -1, T, V)
    return output  # (B, C, T, V)


def type_3_partition(x: Tensor, partition_size: tuple[int, int]):
    '''(B, C, T, V) => (B * N * K, M, L, C)

    Parameters
    - partition_size = (M distant frames, L joints)
        - There are N distant segments and K joint-groups.
        - T = M * N
        - V = K * L
    '''
    B, C, T, V = x.shape
    M, L = partition_size
    partitions = x.view(B, C, M, T // M, V // L, L)
    partitions = partitions.permute(0, 3, 4, 2, 5, 1).contiguous().view(-1, M, L, C)
    return partitions  # (B*N*K, M, L, C)


def type_3_reverse(
    partitions: Tensor,
    original_size: tuple[int, int],
    partition_size: tuple[int, int]
):
    '''(B * N * K, M, L, C) => (B, C, T, V)

    Parameters
    - original_size = (T, V)
    - partition_size = (M distant frames, L joints)
        - There are N distant segments and K joint-groups.
        - T = M * N
        - V = K * L
    '''
    T, V = original_size
    M, L = partition_size
    N, K = T // M, V // L
    B = partitions.shape[0] // (N * K)
    output = partitions.view(B, N, K, M, L, -1)
    output = output.permute(0, 5, 3, 1, 2, 4).contiguous().view(B, -1, T, V)
    return output  # (B, C, T, V)


def type_4_partition(x: Tensor, partition_size: tuple[int, int]):
    '''(B, C, T, V) => (B * N * L, M, K, C)

    Parameters
    - partition_size = (M distant frames, K distant joints)
        - There are N distant segments and L distant joint-groups.
        - T = M * N
        - V = K * L
    '''
    B, C, T, V = x.shape
    M, K = partition_size
    partitions = x.view(B, C, M, T // M, K, V // K)
    partitions = partitions.permute(0, 3, 5, 2, 4, 1).contiguous().view(-1, M, K, C)
    return partitions  # (B*N*L, M, K, C)


def type_4_reverse(
    partitions: Tensor,
    original_size: tuple[int, int],
    partition_size: tuple[int, int]
):
    '''(B * N * L, M, K, C) => (B, C, T, V)

    Parameters
    - original_size = (T, V)
    - partition_size = (M distant frames, K distant joints)
        - There are N distant segments and L distant joint-groups.
        - T = M * N
        - V = K * L
    '''
    T, V = original_size
    M, K = partition_size
    N, L = T // M, V // K
    B = partitions.shape[0] // (N * L)
    output = partitions.view(B, N, L, M, K, -1)
    output = output.permute(0, 5, 3, 1, 4, 2).contiguous().view(B, -1, T, V)
    return output  # (B, C, T, V)


def get_relative_position_index_1d(T):
    '''1D relative positional bias: B_{h}^{t}'''
    coords = torch.stack(torch.meshgrid([torch.arange(T)], indexing='ij'))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += T - 1
    return relative_coords.sum(-1)


class MSA(nn.Module):
    def __init__(
        self,
        rel_type,
        num_heads=32,
        partition_size=(1, 1),
        attn_drop=0.,
        rel=True
    ):
        super().__init__()
        self.rel_type = rel_type
        self.rel = rel
        self.num_heads = num_heads
        self.partition_size = partition_size
        self.scale = num_heads ** -0.5
        self.attn_area = partition_size[0] * partition_size[1]
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.softmax = nn.Softmax(dim=-1)

        if rel:
            match rel_type:
                case 1 | 3:  # attn between neighbor joints
                    self.relative_position_bias_table = nn.Parameter(
                        # 時間軸相對距離關係
                        # Ex: N = 4, dis_type_range = [-3, 3]
                        # => 2N - 1 = 7 types of distance
                        torch.zeros((2 * partition_size[0] - 1), num_heads)
                    )
                case 2 | 4:  # attn between distant joints
                    self.relative_position_bias_table = nn.Parameter(
                        # 時間軸相對距離關係
                        # Ex: N = 4, dis_type_range = [-3, 3]
                        # => 2N - 1 = 7 types of distance
                        torch.zeros((2 * partition_size[0] - 1), partition_size[1], partition_size[1], num_heads)
                        # 因為 distant joints 之間會隱含一些身體部位關係的相同資訊
                        # 但是還是有一些不同的地方需要利用 absolute position bias 區分
                    )
                case _:
                    raise ValueError

            self.register_buffer("relative_position_index", get_relative_position_index_1d(partition_size[0]))
            nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def _get_relative_positional_bias(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        
        match self.rel_type:
            case 1 | 3:
                relative_position_bias = relative_position_bias.view(self.partition_size[0], self.partition_size[0], -1)
                # (T', T', H)
                relative_position_bias = relative_position_bias.unsqueeze(1).unsqueeze(3).repeat(1, self.partition_size[1], 1, self.partition_size[1], 1, 1)
                # (T', V', T', V', H)
            case 2 | 4:
                relative_position_bias = relative_position_bias.view(self.partition_size[0], self.partition_size[0], self.partition_size[1], self.partition_size[1], -1)
                relative_position_bias = relative_position_bias.permute(0, 2, 1, 3, 4)
                # (T', V', T', V', H)
        relative_position_bias = relative_position_bias.contiguous().view(self.attn_area, self.attn_area, -1)
        # (T'V', T'V', H)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)  # (1, H, T'V', T'V')

    def forward(self, x: Tensor):
        B_, S, C = x.shape  # S = T'V'
        qkv = x.view(B_, S, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()
        # qkv: (3, B_, H, S, d=C/H)
        q, k, v = qkv.unbind(0)
        k_T = k.transpose(-2, -1).contiguous()

        dots: Tensor = (q @ k_T) * self.scale
        if self.rel:
            dots = dots + self._get_relative_positional_bias()
        coef: Tensor = self.softmax(dots)
        coef = self.attn_drop(coef)
        # coef: (B_, H, S, S)

        attn = coef @ v  # attn: (B_, H, S, d)
        out = attn.transpose(1, 2).contiguous().view(B_, S, -1)
        return out  # (B_, S, C=H*d)


class SkateFormerBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        num_points=50,
        num_heads=32,
        kernel_size=7,
        type_1_size=(1, 1),
        type_2_size=(1, 1),
        type_3_size=(1, 1),
        type_4_size=(1, 1),
        rel=True,
        attn_drop=0.,
        drop=0.,
        drop_path=0.,
        mlp_ratio=4.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm
):
        super().__init__()
        self.partition_size = [type_1_size, type_2_size, type_3_size, type_4_size]
        self.partition_function = [type_1_partition, type_2_partition, type_3_partition, type_4_partition]
        self.reverse_function = [type_1_reverse, type_2_reverse, type_3_reverse, type_4_reverse]
        self.rel_type = list(range(1, 5))

        self.norm_1 = norm_layer(in_channels)
        self.lin_front = nn.Linear(in_channels, 2 * in_channels, bias=True)
        
        # G-Conv
        self.gconv_w = nn.Parameter(torch.zeros(num_heads // 4, num_points, num_points))
        nn.init.trunc_normal_(self.gconv_w, std=.02)
        
        # T-Conv
        self.tconv = nn.Conv2d(
            in_channels // 4,
            in_channels // 4,
            kernel_size=(kernel_size, 1),
            padding=((kernel_size - 1) // 2, 0),
            groups=num_heads // 4
        )

        # Attention layers
        attention = []
        for i in range(len(self.partition_function)):
            attention.append(MSA(
                rel_type=self.rel_type[i],
                num_heads=num_heads // (len(self.partition_function) * 2),
                partition_size=self.partition_size[i],
                attn_drop=attn_drop,
                rel=rel
            ))
        self.attention = nn.ModuleList(attention)
        
        self.lin_end = nn.Linear(in_channels, in_channels, bias=True)
        self.lin_end_drop = nn.Dropout(p=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Feed Forward
        self.norm_2 = norm_layer(in_channels)
        self.mlp = Mlp(
            in_channels,
            hidden_features=int(mlp_ratio * in_channels),
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x: Tensor):
        B, C, T, V = x.shape

        x = x.permute(0, 2, 3, 1).contiguous()
        skip = x  # skip: (B, T, V, C)

        f: Tensor = self.lin_front(self.norm_1(x))
        f = f.permute(0, 3, 1, 2).contiguous()
        # f: (B, 2C, T, V)

        f_conv, f_attn = torch.split(f, [C // 2, 3 * C // 2], dim=1)
        y = []
        # f_conv: (B, C/2, T, V)
        # f_attn: (B, 3C/2, T, V). Why 3C ? because it's for qkv

        ## G-Conv
        split_f_conv = torch.chunk(f_conv, 2, dim=1)
        # split_f_conv = (B, C/4, T, V), (B, C/4, T, V)

        y_gconv = []
        split_f_gconv = torch.chunk(split_f_conv[0], self.gconv_w.shape[0], dim=1)
        # split_f_gconv = [(B, C/H, T, V)] * (H/4)
        for i in range(self.gconv_w.shape[0]):
            g = torch.einsum('b c t v, v u -> b c t u', split_f_gconv[i], self.gconv_w[i])
            y_gconv.append(g)
        y.append(torch.cat(y_gconv, dim=1))
        # y: [(B, C/4, T, V)]

        ## T-Conv
        y.append(self.tconv(split_f_conv[1]))
        # y: [(B, C/4, T, V)] * 2

        ## Skate-MSA
        split_f_attn = torch.chunk(f_attn, len(self.partition_function), dim=1)
        # split_f_attn: [(B, 3C/8, T, V)] * 4

        for i in range(len(self.partition_function)):
            C = split_f_attn[i].shape[1]  # C_here = 3C/8
            x_partitioned = self.partition_function[i](split_f_attn[i], self.partition_size[i])
            x_partitioned = x_partitioned.view(-1, self.partition_size[i][0] * self.partition_size[i][1], C)
            # x_partitioned: (B_, T'V', C_here = 3C/8)
            y.append(self.reverse_function[i](self.attention[i](x_partitioned), (T, V), self.partition_size[i]))

        y = torch.cat(y, dim=1).permute(0, 2, 3, 1).contiguous()
        # y: (B, T, V, C)

        z = self.lin_end(y)
        z = self.lin_end_drop(z)
        z = self.drop_path(z) + skip

        ## Feed Forward
        out: Tensor = self.drop_path(self.mlp(self.norm_2(z))) + z
        out = out.permute(0, 3, 1, 2).contiguous()
        return out  # (B, C, T, V)


class DownSamplingTempConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=7, stride=2, dilation=1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.reduction = nn.Conv2d(
            in_dim, out_dim,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1)
        )
        self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.bn(self.reduction(x))
        return x


class SkateFormerStage(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        depth,
        first_depth=False,
        num_points=50,
        num_heads=32,
        kernel_size=7,
        t_down_sample_stride=2,
        type_1_size=(1, 1),
        type_2_size=(1, 1),
        type_3_size=(1, 1),
        type_4_size=(1, 1),
        rel=True,
        attn_drop=0.,
        drop=0.,
        drop_path=0.,
        mlp_ratio=4.,
        act_layer=nn.GELU,
        norm_layer_transformer=nn.LayerNorm
    ):
        super().__init__()

        self.ds_tconv = DownSamplingTempConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=t_down_sample_stride
        ) if not first_depth else nn.Identity()

        blocks = []
        for i in range(depth):
            blocks.append(SkateFormerBlock(
                in_channels=in_channels if (i == 0 and first_depth) else out_channels,
                num_points=num_points,
                num_heads=num_heads,
                kernel_size=kernel_size,
                type_1_size=type_1_size,
                type_2_size=type_2_size,
                type_3_size=type_3_size,
                type_4_size=type_4_size,
                rel=rel,
                attn_drop=attn_drop,
                drop=drop,
                drop_path=drop_path if isinstance(drop_path, float) else drop_path[i],
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                norm_layer=norm_layer_transformer
            ))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor):
        x = self.ds_tconv(x)
        for block in self.blocks:
            x = block(x)
        return x


class SkateFormer(nn.Module):
    def __init__(
        self,
        in_channels=3,
        channels=(96, 192, 192, 192),
        depths=(2, 2, 2, 2),
        num_classes=60,
        num_frames=64,
        num_points=50,
        num_people=2,
        num_heads=32,
        kernel_size=7,
        t_down_sample_stride=2,
        type_1_size=(1, 1),
        type_2_size=(1, 1),
        type_3_size=(1, 1),
        type_4_size=(1, 1),
        rel=True,
        attn_drop=0.5,
        drop=0.,
        drop_path=0.2,
        head_drop=0.,
        mlp_ratio=4.,
        act_layer=nn.GELU,
        norm_layer_transformer=nn.LayerNorm,
        use_index_t=False,
        global_pool='avg'
    ):
        super().__init__()

        assert len(depths) == len(channels), "For each stage a channel dimension must be given."
        assert global_pool in ["avg", "max"], f"Only avg and max is supported but {global_pool} is given"
        
        self.num_classes: int = num_classes
        self.use_index_t = use_index_t
        self.proj_dim = channels[0]

        # Projection
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, 2 * in_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            act_layer(),
            nn.Conv2d(2 * in_channels, 3 * in_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            act_layer(),
            nn.Conv2d(3 * in_channels, self.proj_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        )

        # Skate-Embedding
        if use_index_t:
            self.joint_person_embedding = nn.Parameter(
                torch.zeros(self.proj_dim, num_people * num_points)
            )
            nn.init.trunc_normal_(self.joint_person_embedding, std=.02)
        else:
            self.joint_person_temporal_embedding = nn.Parameter(
                torch.zeros(1, self.proj_dim, num_frames, num_people * num_points)
            )
            nn.init.trunc_normal_(self.joint_person_temporal_embedding, std=.02)

        # SkateFormerBlocks + DownSampling T-Convs
        drop_path = torch.linspace(0.0, drop_path, sum(depths)).tolist()
        stages = []
        for i, (channel, depth) in enumerate(zip(channels, depths)):
            stages.append(SkateFormerStage(
                in_channels=channels[0] if i == 0 else channels[i - 1],
                out_channels=channel,
                depth=depth,
                first_depth=i == 0,
                num_points=num_people * num_points,  # m*V
                num_heads=num_heads,
                kernel_size=kernel_size,
                t_down_sample_stride=t_down_sample_stride,
                type_1_size=type_1_size,
                type_2_size=type_2_size,
                type_3_size=type_3_size,
                type_4_size=type_4_size,
                attn_drop=attn_drop,
                drop=drop,
                rel=rel,
                drop_path=drop_path[sum(depths[:i]):sum(depths[:i + 1])],
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                norm_layer_transformer=norm_layer_transformer
            ))
        self.stages = nn.ModuleList(stages)

        # Head
        self.global_pool: str = global_pool
        self.dropout = nn.Dropout(p=head_drop)
        self.lin_head = nn.Linear(channels[-1], num_classes)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = set()
        for n, _ in self.named_parameters():
            if "relative_position_bias_table" in n:
                nwd.add(n)
        return nwd

    def forward_main(self, x: Tensor):
        for stage in self.stages:
            x = stage(x)
        return x

    def forward_head(self, x: Tensor, pre_logits=False):
        if self.global_pool == "avg":
            x = x.mean(dim=(2, 3))
        elif self.global_pool == "max":
            x = torch.amax(x, dim=(2, 3))
        x = self.dropout(x)
        return x if pre_logits else self.lin_head(x)

    def forward(self, x: Tensor, index_t: Tensor = None):
        B, C, T, V, m = x.shape  # m is num people here
        x = x.transpose(3, 4).contiguous().view(B, C, T, -1)
        # x: (B, C, T, m*V)

        x = self.project(x)
        # x: (B, proj_dim, T, m*V)

        # Skate-Embedding
        if self.use_index_t:
            te = torch.zeros(B, T, self.proj_dim).to(x.device)
            div_term = torch.exp(
                torch.arange(0, self.proj_dim, 2, dtype=torch.float) * -(math.log(10000.0) / self.proj_dim)
            ).to(x.device)
            te[:, :, 0::2] = torch.sin(index_t.unsqueeze(-1).float() * div_term)
            te[:, :, 1::2] = torch.cos(index_t.unsqueeze(-1).float() * div_term)
            # te: temporal absolute positional embedding
            x = x + torch.einsum('b t c, c v -> b c t v', te, self.joint_person_embedding).contiguous()
        else:
            x = x + self.joint_person_temporal_embedding

        x = self.forward_main(x)  # x: (B, C', T, m*V)
        x = self.forward_head(x)  # x: (B, num_class)
        return x


if __name__ == "__main__":
    B, C, T, V, m = 3, 2, 32, 25, 2
    N, L = 4, 5  # (N frames, L joints)
    input_size = torch.Size([B, C, T, V, m])

    md = SkateFormer(
        in_channels=2,
        channels=(96, 192, 192),
        depths=(2, 2, 2),
        # Down sampling D=(len(depths)-1) times, if stride is 2
        # => T_final = T / 2^D
        # => T_final still has to be divisible by N and M
        # => You should pick T and N in the following constraints:
        # <<< T = N*M and T_final % N == 0 and T_final % M == 0 >>>
        # If 2^D = 2, min(T) = 4 = 2*2
        # If 2^D = 4, min(T) = 16 = 4*4
        # If 2^D = 8, min(T) = 64 = 8*8
        # IF 2^D = 16, min(T) = 256 = 16*16
        num_classes=35,
        num_frames=T,
        num_points=V,
        num_people=m,
        num_heads=32,
        kernel_size=7,
        type_1_size=(N, L),  # (N frames, L joints)
        type_2_size=(N, V // L),  # (N frames, K distant joints)
        type_3_size=(T // N, L),  # (M distant frames, L joints)
        type_4_size=(T // N, V // L),  # (M distant frames, K distant joints)
        use_index_t=True
    )

    input_data = [torch.zeros(input_size), torch.arange(1, T+1)]
    # out = md(*input_data)
    summary(md, input_data=input_data, depth=3, device='cpu')
