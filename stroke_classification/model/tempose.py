# TemPose: a new skeleton-based transformer model designed for fine-grained motion recognition in badminton
# (2023/08) https://ieeexplore.ieee.org/document/10208321
# Authors: Magnus Ibh, Stella Grasshof, Dan Witzner, Pascal Madeleine

# Modified by Jing-Yuan Chang

import torch
from torch import nn, Tensor
from positional_encodings.torch_encodings import PositionalEncoding1D
from torchinfo import summary


class MLP(nn.Module):
    '''Same as MLP_Block in TemPose paper.'''
    def __init__(self, in_dim, out_dim, hd_dim, drop_p=0.0) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hd_dim),
            nn.GELU(),
            nn.Dropout(drop_p, inplace=True),
            nn.Linear(hd_dim, out_dim)
        )

    def forward(self, x: Tensor):
        return self.mlp(x)


class MLP_Head(nn.Module):
    '''Same as MLP_Head in TemPose.'''
    def __init__(self, in_dim, out_dim, hd_dim, drop_p=0.0) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim)
        self.mlp = MLP(in_dim, out_dim, hd_dim, drop_p)

    def forward(self, x: Tensor):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


class FeedForward(nn.Module):
    '''Same as FeedForward in TemPose.'''
    def __init__(self, in_dim, out_dim, hd_dim, drop_p=0.0) -> None:
        super().__init__()
        self.mlp = MLP(in_dim, out_dim, hd_dim, drop_p)
        self.dropout = nn.Dropout(drop_p, inplace=True)

    def forward(self, x: Tensor):
        x = self.mlp(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    '''Same as Attention in TemPose.'''
    def __init__(self, d_model, d_head, n_head, drop_p) -> None:
        super().__init__()
        d_cat = d_head * n_head

        self.h = n_head
        self.to_qkv = nn.Linear(d_model, d_cat * 3, bias=False)
        self.scale = d_head**-0.5

        self.attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(drop_p)  # This shouldn't be inplace.
        )
        
        self.tail = nn.Sequential(
            nn.Linear(d_cat, d_model),
            nn.Dropout(drop_p, inplace=True)
        ) if n_head != 1 or d_cat != d_model else nn.Identity()

    def forward(self, x: Tensor, mask: Tensor = None):
        # x: (b*n, t, d_model)
        bn, t, _ = x.shape

        qkv: Tensor = self.to_qkv(x)
        qkv = qkv.view(bn, t, self.h, -1).chunk(3, dim=-1)
        q, k, v = map(lambda ts: ts.transpose(1, 2), qkv)
        # q, k, v: (bn, h, t, d_head)

        dots: Tensor = (q.contiguous() @ k.transpose(-1, -2).contiguous()) * self.scale
        # dots: (bn, h, t, t)
        if mask is not None:
            # mask: (bn, t)
            mask = mask.view(bn, 1, 1, t)
            dots = dots.masked_fill(mask == 0.0, -torch.inf)
        
        coef = self.attend(dots)
        attension: Tensor = coef @ v.contiguous()
        # attension: (bn, h, t, d_head)
        
        out = attension.transpose(1, 2).reshape(bn, t, -1)
        # out: (bn, t, h*d_head)
        out = self.tail(out)
        return out  # (bn, t, d_model)


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_head, n_head, hd_mlp, drop_p) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, d_head, n_head, drop_p)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_model, hd_mlp, drop_p)

    def forward(self, x: Tensor, mask=None):
        z = self.layer_norm1(x)
        x = self.attn(z, mask) + x
        z = self.layer_norm2(x)
        x = self.ff(z) + x
        return x


class TransformerEncoder(nn.Module):
    '''Same as Transformer in TemPose.'''
    def __init__(self, d_model, d_head, n_head, depth, hd_mlp, drop_p) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerLayer(d_model, d_head, n_head, hd_mlp, drop_p)
             for _ in range(depth)]
        )

    def forward(self, x: Tensor, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TCN(nn.Module):
    '''Same as TCN in TemPose. There is a bit different from the original TCN.'''
    def __init__(self, in_channel, channels: list[int], kernel_size=5, drop_p=0.3) -> None:
        '''`kernel_size` should be an odd number, so the output sequence length can remain the same as input.'''
        super().__init__()
        layers = []
        for i in range(len(channels)):
            in_ch = in_channel if i == 0 else channels[i-1]
            out_ch = channels[i]
            
            dilation = i * 2 + 1
            padding = (kernel_size - 1) * dilation // 2
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(drop_p, inplace=True)
            ]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: Tensor):
        return self.net(x)


class TemPose_V(nn.Module):
    '''Similar to TemPose_TF in TemPose.'''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4
    ):
        super().__init__()

        self.project = nn.Linear(in_dim, d_model)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+n_people, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # MLP Head
        self.mlp_head = MLP_Head(d_model, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,  # JnB: (b, t, n, input_dim)
        video_len: Tensor  # video_len: (b)
    ):
        JnB = JnB.transpose(1, 2).contiguous()
        # JnB: (b, n, t, input_dim)
        
        x = self.project(JnB)
        b, n, t, d = x.shape

        # Concat cls token (temporal)
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1

        # Temporal embedding
        x = x + self.embedding_tem
        x: Tensor = self.pre_dropout(x)

        # Temporal TransformerLayers
        x = x.view(b*n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, t)
        mask = mask.repeat_interleave(n, dim=0)
        # mask: (b*n, t)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Concat cls token (interactional)
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1

        # Interactional embedding
        x = x + self.embedding_inter

        # Interactional TransformerLayers
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()

        x = self.mlp_head(x)
        return x


class TemPose_PF(nn.Module):
    '''For ablation studies.

    Equal to TemPose_TF without the shuttlecock trajectory
    or TemPose_V with the player positions.
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        '''`d_model` should be an even number.'''
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.project = nn.Linear(in_dim, d_model)

        # TCNs
        tcn_channels = [d_model // 2, d_model]
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people+2, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+n_people+2, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # MLP Head
        self.mlp_head = MLP_Head(d_model, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,  # JnB: (b, t, n, input_dim)
        pos: Tensor,  # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):
        JnB = JnB.transpose(1, 2).contiguous()
        # JnB: (b, n, t, input_dim)
        
        x = self.project(JnB)
        b, n, t, d = x.shape

        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        # pos_top: (b, 2, t)
        # pos_bottom: (b, 2, t)

        # TCNs
        pos_top: Tensor = self.tcn_top(pos_top)
        pos_bottom: Tensor = self.tcn_bottom(pos_bottom)
        # pos_top: (b, d, t)
        # pos_bottom: (b, d, t)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        x_additional = torch.stack((pos_top, pos_bottom), dim=1)
        # x_additional: (b, 2, t, d)

        # Positions Fusion (PF)
        x = torch.cat((x, x_additional), dim=1)
        n += 2

        # Concat cls token (temporal)
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1

        # Temporal embedding
        x = x + self.embedding_tem
        x: Tensor = self.pre_dropout(x)

        # Temporal TransformerLayers
        x = x.view(b*n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, t)
        mask = mask.repeat_interleave(n, dim=0)
        # mask: (b*n, t)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Concat cls token (interactional)
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1

        # Interactional embedding
        x = x + self.embedding_inter

        # Interactional TransformerLayers
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()

        x = self.mlp_head(x)
        return x


class TemPose_SF(nn.Module):
    '''For ablation studies.

    Equal to TemPose_TF without the player positions
    or TemPose_V with the shuttlecock trajectory.
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        '''`d_model` should be an even number.'''
        super().__init__()

        self.project = nn.Linear(in_dim, d_model)

        # TCNs
        tcn_channels = [d_model // 2, d_model]
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people+1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+n_people+1, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # MLP Head
        self.mlp_head = MLP_Head(d_model, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,  # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        video_len: Tensor  # video_len: (b)
    ):
        JnB = JnB.transpose(1, 2).contiguous()
        # JnB: (b, n, t, input_dim)
        
        x = self.project(JnB)
        b, n, t, d = x.shape

        shuttle = shuttle.transpose(1, 2).contiguous()
        # shuttle: (b, 2, t)

        # TCN
        shuttle: Tensor = self.tcn_shuttle(shuttle)
        # shuttle: (b, d, t)

        shuttle = shuttle.transpose(1, 2).contiguous()
        x_additional = shuttle.unsqueeze(1)
        # x_additional: (b, 1, t, d)

        # Shuttlecock Fusion (SF)
        x = torch.cat((x, x_additional), dim=1)
        n += 1

        # Concat cls token (temporal)
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1

        # Temporal embedding
        x = x + self.embedding_tem
        x: Tensor = self.pre_dropout(x)

        # Temporal TransformerLayers
        x = x.view(b*n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, t)
        mask = mask.repeat_interleave(n, dim=0)
        # mask: (b*n, t)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Concat cls token (interactional)
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1

        # Interactional embedding
        x = x + self.embedding_inter

        # Interactional TransformerLayers
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()

        x = self.mlp_head(x)
        return x


class TemPose_TF(nn.Module):
    '''Similar to TemPose_TF in TemPose.'''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        '''`d_model` should be an even number.'''
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.project = nn.Linear(in_dim, d_model)

        # TCNs
        tcn_channels = [d_model // 2, d_model]
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people+3, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+n_people+3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # MLP Head
        self.mlp_head = MLP_Head(d_model, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,  # JnB: (b, t, n, input_dim)
        pos: Tensor,  # pos: (b, t, n, 2)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        video_len: Tensor  # video_len: (b)
    ):
        JnB = JnB.transpose(1, 2).contiguous()
        # JnB: (b, n, t, input_dim)
        
        x = self.project(JnB)
        b, n, t, d = x.shape

        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()
        # pos_top: (b, 2, t)
        # pos_bottom: (b, 2, t)
        # shuttle: (b, 2, t)

        # TCNs
        pos_top: Tensor = self.tcn_top(pos_top)
        pos_bottom: Tensor = self.tcn_bottom(pos_bottom)
        shuttle: Tensor = self.tcn_shuttle(shuttle)
        # pos_top: (b, d, t)
        # pos_bottom: (b, d, t)
        # shuttle: (b, d, t)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)
        # x_additional: (b, 3, t, d)

        # Temporal Fusion (TF)
        x = torch.cat((x, x_additional), dim=1)
        n += 3

        # Concat cls token (temporal)
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1

        # Temporal embedding
        x = x + self.embedding_tem
        x: Tensor = self.pre_dropout(x)

        # Temporal TransformerLayers
        x = x.view(b*n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, t)
        mask = mask.repeat_interleave(n, dim=0)
        # mask: (b*n, t)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Concat cls token (interactional)
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1

        # Interactional embedding
        x = x + self.embedding_inter

        # Interactional TransformerLayers
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()

        x = self.mlp_head(x)
        return x


if __name__ == '__main__':
    b, t, n = 3, 30, 2
    n_features = (17 + 19 * 1) * n
    pose = torch.randn((b, t, n, n_features), dtype=torch.float)
    pos = torch.randn((b, t, n, 2), dtype=torch.float)
    shuttle = torch.randn((b, t, 2), dtype=torch.float)
    videos_len = torch.tensor([t], dtype=torch.long).repeat(b)
    input_data = [pose, pos, shuttle, videos_len]
    model = TemPose_TF(
        in_dim=n_features,
        seq_len=t,
        n_class=35,
        d_model=100
    )
    # input_data = [pose, videos_len]
    # model = TemPose_V(
    #     in_dim=n_features,
    #     seq_len=30,
    #     n_class=35,
    #     d_model=100
    # )
    summary(model, input_data=input_data, depth=4, device='cpu')
