# Writen by Jing-Yuan Chang

import torch
from torch import nn, Tensor
from positional_encodings.torch_encodings import PositionalEncoding1D
from torchinfo import summary

import sys
import os
if __name__ == '__main__':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.tempose import TCN, FeedForward, MLP, MLP_Head, TransformerEncoder


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, d_head, n_head, drop_p) -> None:
        super().__init__()
        d_cat = d_head * n_head

        self.h = n_head
        self.to_q = nn.Linear(d_model, d_cat, bias=False)
        self.to_kv = nn.Linear(d_model, d_cat * 2, bias=False)
        self.scale = d_head**-0.5

        self.attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(drop_p)  # This shouldn't be inplace.
        )
        
        self.tail = nn.Sequential(
            nn.Linear(d_cat, d_model),
            nn.Dropout(drop_p, inplace=True)
        ) if n_head != 1 or d_cat != d_model else nn.Identity()

    def forward(self, x1: Tensor, x2: Tensor, mask: Tensor = None):
        # x1, x2: (b, t, d_model)
        q: Tensor = self.to_q(x1)
        kv: Tensor = self.to_kv(x2)
        b, t, _ = q.shape

        q = q.view(b, t, self.h, -1).transpose(1, 2)
        kv = kv.view(b, t, self.h, -1).chunk(2, dim=-1)
        k, v = map(lambda ts: ts.transpose(1, 2), kv)
        # q, k, v: (b, h, t, d_head)

        dots: Tensor = (q.contiguous() @ k.transpose(-1, -2).contiguous()) * self.scale
        # dots: (b, h, t, t)
        if mask is not None:
            # mask: (b, t)
            mask = mask.view(b, 1, 1, t)
            dots = dots.masked_fill(mask == 0.0, -torch.inf)
        
        coef = self.attend(dots)
        attension: Tensor = coef @ v.contiguous()
        # attension: (b, h, t, d_head)

        out = attension.transpose(1, 2).reshape(b, t, -1)
        # out: (b, t, h*d_head)
        out = self.tail(out)
        return out  # (b, t, d_model)


class CrossTransformerLayer(nn.Module):
    def __init__(self, d_model, d_head, n_head, hd_mlp, drop_p) -> None:
        super().__init__()
        self.layer_norm1_x1 = nn.LayerNorm(d_model)
        self.layer_norm1_x2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadCrossAttention(d_model, d_head, n_head, drop_p)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_model, hd_mlp, drop_p)

    def forward(self, x1: Tensor, x2: Tensor, mask=None):
        x1 = self.layer_norm1_x1(x1)
        x2 = self.layer_norm1_x2(x2)
        x = self.cross_attn(x1, x2, mask)
        z = self.layer_norm2(x)
        x = self.ff(z) + x
        return x


class BST_0(nn.Module):
    '''BST-backbone'''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, [d_model // 2, d_model], tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)
        
        # MLP Head
        self.mlp_head = MLP_Head(d_model * 3, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)
        self.embedding_tem.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)
        self.embedding_cross.copy_(pos_encoding)

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
        b, t, n, in_dim = JnB.shape
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        JnB = self.tcn_pose(JnB)
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)

        shuttle = shuttle.transpose(1, 2).contiguous()
        shuttle = self.tcn_shuttle(shuttle)
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)
        
        x = torch.cat((JnB, shuttle), dim=1)
        _, n, _, d = x.shape

        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)
        x = x.view(b*n, t, d)
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem

        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, 1+t)
        mask_n = mask.repeat_interleave(n, dim=0)
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)
        x = self.encoder_tem(x, mask_n)
        x = x.view(b, n, 1+t, d)

        p1, p2, shuttle = map(lambda ts: ts.squeeze(1), x.chunk(3, dim=1))
        
        p1_cls, p2_cls, shuttle_cls = \
            p1[:, 0].contiguous(), p2[:, 0].contiguous(), shuttle[:, 0].contiguous()

        p1 = p1[:, 1:].contiguous() + self.embedding_cross
        p2 = p2[:, 1:].contiguous() + self.embedding_cross
        shuttle = shuttle[:, 1:].contiguous() + self.embedding_cross

        cross_mask = mask[:, 1:].contiguous()
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)

        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        p1_shuttle = torch.cat((class_token_inter, p1_shuttle), dim=1) + self.embedding_inter
        p2_shuttle = torch.cat((class_token_inter, p2_shuttle), dim=1) + self.embedding_inter

        p1_shuttle: Tensor = self.encoder_inter(p1_shuttle, mask)
        p2_shuttle: Tensor = self.encoder_inter(p2_shuttle, mask)

        p1_shuttle_cls = p1_shuttle[:, 0, :].contiguous()
        p2_shuttle_cls = p2_shuttle[:, 0, :].contiguous()

        p1_conclusion = p1_cls + p1_shuttle_cls
        p2_conclusion = p2_cls + p2_shuttle_cls

        x = torch.cat((p1_conclusion, p2_conclusion, shuttle_cls), dim=1)
        x = self.mlp_head(x)
        return x


class BST(nn.Module):
    '''BST
    - PPF: Pose Position Fusion
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.mlp_positions = MLP(2, out_dim=in_dim, hd_dim=256, drop_p=drop_p)

        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, [d_model // 2, d_model], tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)
        
        # MLP Head
        self.mlp_head = MLP_Head(d_model * 3, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)
        self.embedding_tem.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)
        self.embedding_cross.copy_(pos_encoding)

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
        JnB: Tensor,      # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        pos: Tensor,      # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):
        b, t, n, in_dim = JnB.shape
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        
        pos = self.mlp_positions(pos)
        pos_impact = pos.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)

        JnB = JnB * pos_impact + JnB

        JnB = self.tcn_pose(JnB)
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)

        shuttle = shuttle.transpose(1, 2).contiguous()
        shuttle = self.tcn_shuttle(shuttle)
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)
        
        x = torch.cat((JnB, shuttle), dim=1)
        _, n, _, d = x.shape

        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)
        x = x.view(b*n, t, d)
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem

        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, 1+t)
        mask_n = mask.repeat_interleave(n, dim=0)
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)
        x = self.encoder_tem(x, mask_n)
        x = x.view(b, n, 1+t, d)

        p1, p2, shuttle = map(lambda ts: ts.squeeze(1), x.chunk(3, dim=1))
        
        p1_cls, p2_cls, shuttle_cls = \
            p1[:, 0].contiguous(), p2[:, 0].contiguous(), shuttle[:, 0].contiguous()

        p1 = p1[:, 1:].contiguous() + self.embedding_cross
        p2 = p2[:, 1:].contiguous() + self.embedding_cross
        shuttle = shuttle[:, 1:].contiguous() + self.embedding_cross

        cross_mask = mask[:, 1:].contiguous()
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)

        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        p1_shuttle = torch.cat((class_token_inter, p1_shuttle), dim=1) + self.embedding_inter
        p2_shuttle = torch.cat((class_token_inter, p2_shuttle), dim=1) + self.embedding_inter

        p1_shuttle: Tensor = self.encoder_inter(p1_shuttle, mask)
        p2_shuttle: Tensor = self.encoder_inter(p2_shuttle, mask)

        p1_shuttle_cls = p1_shuttle[:, 0, :].contiguous()
        p2_shuttle_cls = p2_shuttle[:, 0, :].contiguous()

        p1_conclusion = p1_cls + p1_shuttle_cls
        p2_conclusion = p2_cls + p2_shuttle_cls

        x = torch.cat((p1_conclusion, p2_conclusion, shuttle_cls), dim=1)
        x = self.mlp_head(x)
        return x


class BST_CG(nn.Module):
    '''BST
    - PPF: Pose Position Fusion
    - Adding Clean Gate
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.mlp_positions = MLP(2, out_dim=in_dim, hd_dim=256, drop_p=drop_p)

        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, [d_model // 2, d_model], tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)
        
        # Clean Gate
        self.mlp_clean = MLP(d_model, d_model, d_model, drop_p)

        # MLP Head
        self.mlp_head = MLP_Head(d_model * 3, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)
        self.embedding_tem.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)
        self.embedding_cross.copy_(pos_encoding)

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
        JnB: Tensor,      # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        pos: Tensor,      # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):
        b, t, n, in_dim = JnB.shape
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        
        pos = self.mlp_positions(pos)
        pos_impact = pos.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)

        JnB = JnB * pos_impact + JnB

        JnB = self.tcn_pose(JnB)
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)

        shuttle = shuttle.transpose(1, 2).contiguous()
        shuttle = self.tcn_shuttle(shuttle)
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)
        
        x = torch.cat((JnB, shuttle), dim=1)
        _, n, _, d = x.shape

        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)
        x = x.view(b*n, t, d)
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem

        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, 1+t)
        mask_n = mask.repeat_interleave(n, dim=0)
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)
        x = self.encoder_tem(x, mask_n)
        x = x.view(b, n, 1+t, d)

        p1, p2, shuttle = map(lambda ts: ts.squeeze(1), x.chunk(3, dim=1))
        
        p1_cls, p2_cls, shuttle_cls = \
            p1[:, 0].contiguous(), p2[:, 0].contiguous(), shuttle[:, 0].contiguous()

        p1 = p1[:, 1:].contiguous() + self.embedding_cross
        p2 = p2[:, 1:].contiguous() + self.embedding_cross
        shuttle = shuttle[:, 1:].contiguous() + self.embedding_cross

        cross_mask = mask[:, 1:].contiguous()
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)

        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        p1_shuttle = torch.cat((class_token_inter, p1_shuttle), dim=1) + self.embedding_inter
        p2_shuttle = torch.cat((class_token_inter, p2_shuttle), dim=1) + self.embedding_inter

        p1_shuttle: Tensor = self.encoder_inter(p1_shuttle, mask)
        p2_shuttle: Tensor = self.encoder_inter(p2_shuttle, mask)

        p1_shuttle_cls = p1_shuttle[:, 0, :].contiguous()
        p2_shuttle_cls = p2_shuttle[:, 0, :].contiguous()

        # Clean Gate
        info_need_clean = torch.minimum(p1_shuttle_cls, p2_shuttle_cls)
        dirt = self.mlp_clean(info_need_clean)
        shuttle_cls = shuttle_cls - dirt

        p1_conclusion = p1_cls + p1_shuttle_cls
        p2_conclusion = p2_cls + p2_shuttle_cls

        x = torch.cat((p1_conclusion, p2_conclusion, shuttle_cls), dim=1)
        x = self.mlp_head(x)
        return x


class BST_AP(nn.Module):
    '''BST_AimPlayer
    - PPF: Pose Position Fusion
    - Adding Cosine Simularity to determine alpha
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.mlp_positions = MLP(2, out_dim=in_dim, hd_dim=256, drop_p=drop_p)

        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, [d_model // 2, d_model], tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)
        
        # Cosine Simularity
        self.cos_sim = nn.CosineSimilarity()

        # MLP Head
        self.mlp_head = MLP_Head(d_model * 2, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)
        self.embedding_tem.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)
        self.embedding_cross.copy_(pos_encoding)

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
        JnB: Tensor,      # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        pos: Tensor,      # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):
        b, t, n, in_dim = JnB.shape
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        
        pos = self.mlp_positions(pos)
        pos_impact = pos.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)

        JnB = JnB * pos_impact + JnB

        JnB = self.tcn_pose(JnB)
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)

        shuttle = shuttle.transpose(1, 2).contiguous()
        shuttle = self.tcn_shuttle(shuttle)
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)
        
        x = torch.cat((JnB, shuttle), dim=1)
        _, n, _, d = x.shape

        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)
        x = x.view(b*n, t, d)
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem

        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, 1+t)
        mask_n = mask.repeat_interleave(n, dim=0)
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)
        x = self.encoder_tem(x, mask_n)
        x = x.view(b, n, 1+t, d)

        p1, p2, shuttle = map(lambda ts: ts.squeeze(1), x.chunk(3, dim=1))
        
        p1_cls, p2_cls, shuttle_cls = \
            p1[:, 0].contiguous(), p2[:, 0].contiguous(), shuttle[:, 0].contiguous()

        p1 = p1[:, 1:].contiguous() + self.embedding_cross
        p2 = p2[:, 1:].contiguous() + self.embedding_cross
        shuttle = shuttle[:, 1:].contiguous() + self.embedding_cross

        cross_mask = mask[:, 1:].contiguous()
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)

        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        p1_shuttle = torch.cat((class_token_inter, p1_shuttle), dim=1) + self.embedding_inter
        p2_shuttle = torch.cat((class_token_inter, p2_shuttle), dim=1) + self.embedding_inter

        p1_shuttle: Tensor = self.encoder_inter(p1_shuttle, mask)
        p2_shuttle: Tensor = self.encoder_inter(p2_shuttle, mask)

        p1_shuttle_cls = p1_shuttle[:, 0, :].contiguous()
        p2_shuttle_cls = p2_shuttle[:, 0, :].contiguous()

        p1_conclusion = p1_cls + p1_shuttle_cls
        p2_conclusion = p2_cls + p2_shuttle_cls

        # Compute Cosine Simularities
        p1_shuttle_sim = self.cos_sim(p1_shuttle_cls, shuttle_cls)
        p2_shuttle_sim = self.cos_sim(p2_shuttle_cls, shuttle_cls)
        alpha: Tensor = (p1_shuttle_sim - p2_shuttle_sim + 2) / 4
        alpha = alpha.unsqueeze(1)

        p1_conclusion = alpha * p1_conclusion
        p2_conclusion = (1-alpha) * p2_conclusion

        x = torch.cat((p1_conclusion, p2_conclusion), dim=1)
        x = self.mlp_head(x)
        return x


class BST_CG_AP(nn.Module):
    '''BST_CleanGate_AimPlayer
    - PPF: Pose Position Fusion
    - Adding Clean Gate and Cosine Simularity
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.mlp_positions = MLP(2, out_dim=in_dim, hd_dim=256, drop_p=drop_p)

        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, [d_model // 2, d_model], tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)
        
        # Cosine Simularity
        self.cos_sim = nn.CosineSimilarity()

        # Clean Gate
        self.mlp_clean = MLP(d_model, d_model, d_model, drop_p)

        # MLP Head
        self.mlp_head = MLP_Head(d_model * 3, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)
        self.embedding_tem.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)
        self.embedding_cross.copy_(pos_encoding)

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
        JnB: Tensor,      # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        pos: Tensor,      # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):
        b, t, n, in_dim = JnB.shape
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        
        pos = self.mlp_positions(pos)
        pos_impact = pos.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)

        JnB = JnB * pos_impact + JnB

        JnB = self.tcn_pose(JnB)
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)

        shuttle = shuttle.transpose(1, 2).contiguous()
        shuttle = self.tcn_shuttle(shuttle)
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)
        
        x = torch.cat((JnB, shuttle), dim=1)
        _, n, _, d = x.shape

        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)
        x = x.view(b*n, t, d)
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem

        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, 1+t)
        mask_n = mask.repeat_interleave(n, dim=0)
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)
        x = self.encoder_tem(x, mask_n)
        x = x.view(b, n, 1+t, d)

        p1, p2, shuttle = map(lambda ts: ts.squeeze(1), x.chunk(3, dim=1))
        
        p1_cls, p2_cls, shuttle_cls = \
            p1[:, 0].contiguous(), p2[:, 0].contiguous(), shuttle[:, 0].contiguous()

        p1 = p1[:, 1:].contiguous() + self.embedding_cross
        p2 = p2[:, 1:].contiguous() + self.embedding_cross
        shuttle = shuttle[:, 1:].contiguous() + self.embedding_cross

        cross_mask = mask[:, 1:].contiguous()
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)

        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        p1_shuttle = torch.cat((class_token_inter, p1_shuttle), dim=1) + self.embedding_inter
        p2_shuttle = torch.cat((class_token_inter, p2_shuttle), dim=1) + self.embedding_inter

        p1_shuttle: Tensor = self.encoder_inter(p1_shuttle, mask)
        p2_shuttle: Tensor = self.encoder_inter(p2_shuttle, mask)

        p1_shuttle_cls = p1_shuttle[:, 0, :].contiguous()
        p2_shuttle_cls = p2_shuttle[:, 0, :].contiguous()

        # Compute Cosine Simularities
        p1_shuttle_sim = self.cos_sim(p1_shuttle_cls, shuttle_cls)
        p2_shuttle_sim = self.cos_sim(p2_shuttle_cls, shuttle_cls)
        alpha: Tensor = (p1_shuttle_sim - p2_shuttle_sim + 2) / 4
        alpha = alpha.unsqueeze(1)

        p1_conclusion = p1_cls + p1_shuttle_cls
        p2_conclusion = p2_cls + p2_shuttle_cls

        p1_conclusion = alpha * p1_conclusion
        p2_conclusion = (1-alpha) * p2_conclusion

        # Clean Gate
        info_need_clean = torch.minimum(p1_shuttle_cls, p2_shuttle_cls)
        dirt = self.mlp_clean(info_need_clean)
        shuttle_cls = shuttle_cls - dirt

        x = torch.cat((p1_conclusion, p2_conclusion, shuttle_cls), dim=1)
        x = self.mlp_head(x)
        return x


if __name__ == '__main__':
    b, t, n = 3, 30, 2
    n_features = (17 + 19 * 1) * n
    pose = torch.randn((b, t, n, n_features), dtype=torch.float)
    shuttle = torch.randn((b, t, 2), dtype=torch.float)
    pos = torch.randn((b, t, n, 2), dtype=torch.float)
    videos_len = torch.tensor([t], dtype=torch.long).repeat(b)
    input_data = [pose, shuttle, pos, videos_len]
    model = BST(
        in_dim=n_features,
        seq_len=t,
        n_class=35,
        d_model=100
    )
    summary(model, input_data=input_data, depth=4, device='cpu')
