# BlockGCN: Redefine Topology Awareness for Skeleton-Based Action Recognition
# (2024 CVPR) https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_BlockGCN_Redefine_Topology_Awareness_for_Skeleton-Based_Action_Recognition_CVPR_2024_paper.html
# Authors: Yuxuan Zhou, Xudong Yan, Zhi-Qi Cheng, Yan Yan, Qi Dai, Xian-Sheng Hua

# Modified by Jing-Yuan Chang

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchinfo import summary

# Use Python < 3.12 to build torch_topological
# And upgrade POT package to >= 0.9.4 after torch_topological installed
from torch_topological.nn import VietorisRipsComplex
from torch_topological.nn.data import make_tensor
from torch_topological.nn.layers import StructureElementLayer

import numpy as np


class Graph:
    def __init__(self, layout='mediapipe', labeling_mode='spatial'):
        match layout:
            case 'mediapipe':
                num_node = 23
                origin_edges = [
                    (2,1),(3,1),  # shoulders to head (for connected graph)
                    (6,12),(6,10),(6,8),(8,10),       # left hand
                    (7,13),(7,11),(7,9),(9,11),       # right hand
                    (2,4),(4,6),(3,5),(5,7),          # arms
                    (2,3),(2,14),(3,15),(14,15),      # torso
                    (14,16),(15,17),(16,18),(17,19),  # legs
                    (18,20),(18,22),(20,22),          # left foot
                    (19,21),(19,23),(21,23)           # right foot
                ]

            case 'coco':
                num_node = 17
                origin_edges = [
                    (16,14),(14,12),(17,15),(15,13),  # legs
                    (12,13),(6,12),(7,13),(6,7),      # torso
                    (8,6),(10,8),(9,7),(11,9),        # arms
                    (2,3),                            # between eyes
                    (2,1),(3,1),(4,2),(5,3),          # head
                    (4,6),(5,7)                       # ears to shoulders
                ]

            case _:
                raise NotImplementedError
        
        self.num_node = num_node
        self.self_link = [(i, i) for i in range(num_node)]
        
        self.fw_edges = [(i-1, j-1) for (i, j) in origin_edges]
        self.bw_edges = [(j, i) for (i, j) in self.fw_edges]
        self.neighbors = self.fw_edges + self.bw_edges
        
        A = self.get_adjacency_matrix(labeling_mode)
        self.hop = self.get_hop_distance(A)

    def edge2mat(self, edges):
        A = np.zeros((self.num_node, self.num_node))
        for j, i in edges:  # j -> i (same behavior to origin)
            A[i, j] = 1
        return A

    def normalize_digraph(self, A: np.ndarray):
        '''Assume j -> i, A_normalized should be at the left side of the input matrix X.'''
        in_degree = np.sum(A, 1)  # modified
        D = np.zeros((self.num_node, self.num_node))
        for i in range(self.num_node):
            if in_degree[i] > 0:
                D[i, i] = in_degree[i]**(-1)
        DA = D @ A  # modified
        return DA

    def get_uniform_graph(self, self_link, neighbors):
        A = self.normalize_digraph(self.edge2mat(self_link + neighbors))
        return A[None, :]  # (1, v, v)

    def get_spatial_graph(self, self_link, fw, bw):
        I = self.edge2mat(self_link)
        Fw = self.normalize_digraph(self.edge2mat(fw))
        Bw = self.normalize_digraph(self.edge2mat(bw))
        A = np.stack((I, Fw, Bw))
        return A  # (3, v, v)

    def get_adjacency_matrix(self, labeling_mode):
        match labeling_mode:
            case 'uniform':
                A = self.get_uniform_graph(self.self_link, self.neighbors)
            case 'spatial':
                A = self.get_spatial_graph(self.self_link, self.fw_edges, self.bw_edges)
            case _:
                raise NotImplementedError
        return A  # (ks, v, v)

    def get_hop_distance(self, A: np.ndarray):
        v = A.shape[-1]
        A = A.sum(0)  # A: (v, v)
        A[A != 0] = 1

        M = [np.eye(v), A]
        for d in range(2, v):
            M.append(M[d-1] @ A)
        
        M = np.stack(M) > 0
        
        hop_dis = np.full((v, v), v, dtype=np.int32)
        for d in range(v-1, -1, -1):
            hop_dis[M[d]] = d

        assert np.all(hop_dis != v), 'The graph should be connected.'
        return hop_dis


def conv_init(conv: nn.Conv2d):
    if isinstance(conv, nn.Conv2d):
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)


def bn_init(m):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight, 1.0, 0.02)


class D_TopoEncoder(nn.Module):
    '''A bit different from the original Topo.'''
    def __init__(self, ph_dims=0, out_features=64, n_people=1):
        super().__init__()
        self.vr = VietorisRipsComplex(dim=ph_dims)
        self.pl = StructureElementLayer(n_elements=out_features)
        self.n_people = n_people

    def L2_norm(self, weight):
        weight_norm = torch.norm(weight, p=2, dim=1)
        return weight_norm

    def forward(self, x: Tensor):
        N, M, C, T, V = x.shape

        x = x.flatten(end_dim=1)  # (n*m, c, t, v)
        x = x.unsqueeze(-1) - x.unsqueeze(-2)  # (n*m, c, t, v, v)
        x = x.mean(-3)  # (n*m, c, v, v)
        x = self.L2_norm(x)  # (n*m, v, v)
        x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))

        x = self.vr(x)
        x = make_tensor(x)
        x = self.pl(x)
        x = x.view(N*M, -1)  # since last layer whould skip the batch dim if batch = 1
        return x  # (n*m, d)


class D_TopoProjector(nn.Module):
    def __init__(self, out_dim, in_dim=64):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: Tensor):
        # (n*m, c)
        x = self.lin(x)
        x = self.bn(x)
        x = self.relu(x)
        return x.view(*x.shape, 1, 1)


class TemporalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, norm=True):
        '''
        Only do convolution on dimension -2 (time) without dimension -1 (last).

        A batch normalization is added to the bottom.
        '''
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1)
        )
        self.bn = nn.BatchNorm2d(out_channels) if norm else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class BlockGC(nn.Module):
    def __init__(self,
        in_channels, out_channels,
        hop: np.ndarray,
        kernel_size=3,
        n_heads=8,
    ):
        super().__init__()
        self.out_c = out_channels

        # K groups in the paper is corresponding to n_heads
        self.k = kernel_size
        self.h = n_heads if in_channels > n_heads else 1

        # To build matrix B in the paper
        self.hop = hop  # It is i -> j here.
        self.emb_table = nn.Parameter(torch.zeros(self.k, self.h, hop.max()+1))

        # Learnable matrix A in the paper
        v, _ = hop.shape
        self.A = nn.Parameter(torch.eye(v).repeat(self.k, self.h, 1, 1))

        # The Weight Matrix in the paper + bias
        self.block_net = nn.Conv2d(
            self.k * in_channels, 
            self.k * out_channels,
            kernel_size=1,
            groups=(self.k * self.h)
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.res = TemporalConv2d(in_channels, out_channels, 1) if in_channels != out_channels else \
                   nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        # self.apply(conv_init)

    def L2_norm(self, weight):
        '''Assume i -> j here and use the in-degree to normalize.'''
        weight_norm = torch.norm(weight, p=2, dim=-2, keepdim=True) + 1e-4
        return weight_norm

    def forward(self, x: Tensor):
        N, C, T, V = x.size()
        res = self.res(x)

        B = self.emb_table[:, :, self.hop]  # B: (k, h, v, v)
        A = self.A                          # A: (k, h, v, v)
        BnA = B / self.L2_norm(B) + A / self.L2_norm(A)

        x = x.view(N, 1, self.h, C // self.h, T, V)
        x = torch.einsum("n k h c t v , k h v w -> n k h c t w", (x, BnA)).contiguous()
        x = x.view(N, -1, T, V)

        x = self.block_net(x)
        x = x.view(N, self.k, -1, T, V)
        x = x.sum(dim=1)

        x = self.bn(x)
        x = self.relu(x + res)
        return x


class MultiScale_TCN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        stride=1,
        dilations=[1, 2],
        residual=False,
        residual_kernel_size=1
    ):
        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'
        if isinstance(kernel_size, list):
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)

        # Multiple branches of temporal convolution
        num_branches = len(dilations) + 2
        branch_ch = out_channels // num_branches
        
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, branch_ch, 1, padding=0),
                nn.BatchNorm2d(branch_ch),
                nn.ReLU(inplace=True),
                TemporalConv2d(
                    branch_ch,
                    branch_ch,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation
                ),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Max branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, 1, padding=0),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_ch)  # 为什么还要加bn <= 原作的註解，我怎麼知道？？
        ))

        # 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, 1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_ch)
        ))

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = nn.Identity()
        else:
            self.residual = TemporalConv2d(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # self.apply(conv_init)
        # self.apply(bn_init)

    def forward(self, x):
        # (N, C, T, V)
        res = self.residual(x)
        out = torch.cat([tcn_branch(x) for tcn_branch in self.branches], dim=1)
        return out + res


class GCN_TCN_Layer(nn.Module):
    def __init__(self,
        in_channels, out_channels, hop, g_kernel_size=3, n_heads=8,
        t_kernel_size=5, stride=1, dilation=1,
        residual=True,
    ):
        super().__init__()
        self.project = D_TopoProjector(in_channels)
        self.gcn = BlockGC(in_channels, out_channels, hop, g_kernel_size, n_heads)
        self.tcn = TemporalConv2d(
            out_channels, out_channels,
            kernel_size=t_kernel_size,
            stride=stride,
            dilation=dilation,
        )
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = nn.Identity()
        else:
            self.residual = TemporalConv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, d_topo):
        d_topo_emb = self.project(d_topo)
        x = x + d_topo_emb

        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = self.relu(x + res)
        return x


class GCN_MultiScale_TCN_Layer(nn.Module):
    def __init__(self,
        in_channels, out_channels, hop, g_kernel_size=3, n_heads=8,
        t_kernel_size=5, stride=1, dilations=[1, 2],
        residual=True,
    ):
        super().__init__()
        self.project = D_TopoProjector(in_channels)
        self.gcn = BlockGC(in_channels, out_channels, hop, g_kernel_size, n_heads)
        self.tcn = MultiScale_TCN(
            out_channels, out_channels,
            kernel_size=t_kernel_size,
            stride=stride,
            dilations=dilations,
            residual=False
        )
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = nn.Identity()
        else:
            self.residual = TemporalConv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, d_topo):
        d_topo_emb = self.project(d_topo)
        x = x + d_topo_emb

        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = self.relu(x + res)
        return x


class BlockGCN_10(nn.Module):
    def __init__(
        self,
        num_class,
        num_person,
        in_channels,
        graph_args,
        g_kernel_size=3,
        n_heads=8,
        t_kernel_size=5,
        data_bn=True,
        last_drop_out=0,
    ):
        super().__init__()
        self.graph = Graph(**graph_args)
        v = self.graph.num_node
        hop = self.graph.hop  # (v, v) If hop isn't symmetric, it should be transposed here.

        self.channel_rise = nn.Linear(in_channels, 128)
        self.point_pos_emb = nn.Parameter(torch.randn(1, v, 128))

        self.data_bn = nn.BatchNorm1d(num_person * v * 128) if data_bn else \
                       nn.Identity()

        self.dynamic_topo_encode = D_TopoEncoder(n_people=num_person)

        self.body = nn.ModuleList([
            GCN_MultiScale_TCN_Layer(128, 128, hop, g_kernel_size, n_heads, t_kernel_size),
            GCN_MultiScale_TCN_Layer(128, 128, hop, g_kernel_size, n_heads, t_kernel_size),
            GCN_MultiScale_TCN_Layer(128, 128, hop, g_kernel_size, n_heads, t_kernel_size),
            GCN_MultiScale_TCN_Layer(128, 128, hop, g_kernel_size, n_heads, t_kernel_size),
            GCN_MultiScale_TCN_Layer(128, 256, hop, g_kernel_size, n_heads, t_kernel_size, stride=2),
            GCN_MultiScale_TCN_Layer(256, 256, hop, g_kernel_size, n_heads, t_kernel_size),
            GCN_MultiScale_TCN_Layer(256, 256, hop, g_kernel_size, n_heads, t_kernel_size),
            GCN_MultiScale_TCN_Layer(256, 256, hop, g_kernel_size, n_heads, t_kernel_size, stride=2),
            GCN_MultiScale_TCN_Layer(256, 256, hop, g_kernel_size, n_heads, t_kernel_size),
            GCN_MultiScale_TCN_Layer(256, 256, hop, g_kernel_size, n_heads, t_kernel_size),
        ])
        
        self.drop_out = nn.Dropout(last_drop_out, inplace=True) if last_drop_out else \
                        nn.Identity()
        self.fc = nn.Linear(256, num_class)

        # nn.init.normal_(self.fc.weight, 0, (2. / num_class)**0.5)

    def forward(self, x: Tensor):
        N, C, T, V, M = x.size()

        d_topo = x.permute(0, 4, 1, 2, 3).contiguous()
        # d_topo: (n, m, c, t, v)
        d_topo = self.dynamic_topo_encode(d_topo)
        # d_topo: (n*m, d)

        x = x.permute(0, 4, 2, 3, 1).reshape(-1, V, C)
        x = self.channel_rise(x)
        x = x + self.point_pos_emb
        # x: (n*m*t, v, c)

        x = x.view(N, M, T, V, -1).permute(0, 1, 3, 4, 2).reshape(N, -1, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, -1, T).permute(0, 1, 3, 4, 2).reshape(N * M, -1, T, V)

        for layer in self.body:
            x = layer(x, d_topo)

        # x: (n*m, c, t, v)
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)
        x = self.fc(x)
        return x


class BlockGCN_6(nn.Module):
    def __init__(
        self,
        num_class,
        num_person,
        in_channels,
        graph_args,
        g_kernel_size=3,
        n_heads=8,
        t_kernel_size=5,
        data_bn=True,
        last_drop_out=0,
    ):
        super().__init__()
        self.graph = Graph(**graph_args)
        v = self.graph.num_node
        hop = self.graph.hop  # (v, v) If hop isn't symmetric, it should be transposed here.

        self.channel_rise = nn.Linear(in_channels, 128)
        self.point_pos_emb = nn.Parameter(torch.randn(1, v, 128))

        self.data_bn = nn.BatchNorm1d(num_person * v * 128) if data_bn else \
                       nn.Identity()

        self.dynamic_topo_encode = D_TopoEncoder(n_people=num_person)

        self.body = nn.ModuleList([
            GCN_MultiScale_TCN_Layer(128, 128, hop, g_kernel_size, n_heads, t_kernel_size),
            GCN_MultiScale_TCN_Layer(128, 128, hop, g_kernel_size, n_heads, t_kernel_size),
            GCN_MultiScale_TCN_Layer(128, 256, hop, g_kernel_size, n_heads, t_kernel_size, stride=2),
            GCN_MultiScale_TCN_Layer(256, 256, hop, g_kernel_size, n_heads, t_kernel_size),
            GCN_MultiScale_TCN_Layer(256, 256, hop, g_kernel_size, n_heads, t_kernel_size, stride=2),
            GCN_MultiScale_TCN_Layer(256, 256, hop, g_kernel_size, n_heads, t_kernel_size),
        ])
        
        self.drop_out = nn.Dropout(last_drop_out, inplace=True) if last_drop_out else \
                        nn.Identity()
        self.fc = nn.Linear(256, num_class)

        # nn.init.normal_(self.fc.weight, 0, (2. / num_class)**0.5)

    def forward(self, x: Tensor):
        N, C, T, V, M = x.size()

        d_topo = x.permute(0, 4, 1, 2, 3).contiguous()
        # d_topo: (n, m, c, t, v)
        d_topo = self.dynamic_topo_encode(d_topo)
        # d_topo: (n*m, d)

        x = x.permute(0, 4, 2, 3, 1).reshape(-1, V, C)
        x = self.channel_rise(x)
        x = x + self.point_pos_emb
        # x: (n*m*t, v, c)

        x = x.view(N, M, T, V, -1).permute(0, 1, 3, 4, 2).reshape(N, -1, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, -1, T).permute(0, 1, 3, 4, 2).reshape(N * M, -1, T, V)

        for layer in self.body:
            x = layer(x, d_topo)

        # x: (n*m, c, t, v)
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)
        x = self.fc(x)
        return x


class BlockGCN_6_normal_TCN_per_frame(nn.Module):
    def __init__(
        self,
        num_class,
        num_person,
        in_channels,
        graph_args,
        g_kernel_size=3,
        n_heads=4,
        t_kernel_size=9,
        data_bn=True,
        last_drop_out=0,
    ):
        super().__init__()
        self.graph = Graph(**graph_args)
        v = self.graph.num_node
        hop = self.graph.hop  # (v, v) If hop isn't symmetric, it should be transposed here.

        self.channel_rise = nn.Linear(in_channels, 64)
        self.point_pos_emb = nn.Parameter(torch.randn(1, v, 64))

        self.data_bn = nn.BatchNorm1d(num_person * v * 64) if data_bn else \
                       nn.Identity()

        self.dynamic_topo_encode = D_TopoEncoder(n_people=num_person)

        self.body = nn.ModuleList([
            GCN_TCN_Layer(64, 64, hop, g_kernel_size, n_heads, t_kernel_size),
            GCN_TCN_Layer(64, 64, hop, g_kernel_size, n_heads, t_kernel_size),
            GCN_TCN_Layer(64, 128, hop, g_kernel_size, n_heads, t_kernel_size),
            GCN_TCN_Layer(128, 128, hop, g_kernel_size, n_heads, t_kernel_size),
            GCN_TCN_Layer(128, 128, hop, g_kernel_size, n_heads, t_kernel_size),
            GCN_TCN_Layer(128, 128, hop, g_kernel_size, n_heads, t_kernel_size),
        ])
        
        self.drop_out = nn.Dropout(last_drop_out, inplace=True) if last_drop_out else \
                        nn.Identity()
        self.fc = nn.Linear(128, num_class)

    def forward(self, x: Tensor):
        N, C, T, V, M = x.size()

        d_topo = x.permute(0, 4, 1, 2, 3).contiguous()
        # d_topo: (n, m, c, t, v)
        d_topo = self.dynamic_topo_encode(d_topo)
        # d_topo: (n*m, d)

        x = x.permute(0, 4, 2, 3, 1).reshape(-1, V, C)
        x = self.channel_rise(x)
        x = x + self.point_pos_emb
        # x: (n*m*t, v, c)

        x = x.view(N, M, T, V, -1).permute(0, 1, 3, 4, 2).reshape(N, -1, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, -1, T).permute(0, 1, 3, 4, 2).reshape(N * M, -1, T, V)

        for layer in self.body:
            x = layer(x, d_topo)

        # x: (n*m, c, t, v)
        c_new = x.size(1)
        x = x.view(N, M, c_new, T, V)
        x = x.mean(-1).mean(1)
        x = self.drop_out(x)
        # x: (n, c, t)
        x = x.transpose(1, 2).contiguous()
        # x: (n, t, c)
        x = self.fc(x)
        return x


class BlockGCN_6_per_frame(nn.Module):
    def __init__(
        self,
        num_class,
        num_person,
        in_channels,
        graph_args,
        g_kernel_size=3,
        n_heads=8,
        t_kernel_size=5,
        data_bn=True,
        last_drop_out=0,
    ):
        super().__init__()
        self.graph = Graph(**graph_args)
        v = self.graph.num_node
        hop = self.graph.hop  # (v, v) If hop isn't symmetric, it should be transposed here.

        self.channel_rise = nn.Linear(in_channels, 128)
        self.point_pos_emb = nn.Parameter(torch.randn(1, v, 128))

        self.data_bn = nn.BatchNorm1d(num_person * v * 128) if data_bn else \
                       nn.Identity()

        self.dynamic_topo_encode = D_TopoEncoder(n_people=num_person)

        self.body = nn.ModuleList([
            GCN_MultiScale_TCN_Layer(128, 128, hop, g_kernel_size, n_heads, t_kernel_size),
            GCN_MultiScale_TCN_Layer(128, 128, hop, g_kernel_size, n_heads, t_kernel_size),
            GCN_MultiScale_TCN_Layer(128, 256, hop, g_kernel_size, n_heads, t_kernel_size),
            GCN_MultiScale_TCN_Layer(256, 256, hop, g_kernel_size, n_heads, t_kernel_size),
            GCN_MultiScale_TCN_Layer(256, 256, hop, g_kernel_size, n_heads, t_kernel_size),
            GCN_MultiScale_TCN_Layer(256, 256, hop, g_kernel_size, n_heads, t_kernel_size),
        ])
        
        self.drop_out = nn.Dropout(last_drop_out, inplace=True) if last_drop_out else \
                        nn.Identity()
        self.fc = nn.Linear(256, num_class)

    def forward(self, x: Tensor):
        N, C, T, V, M = x.size()

        d_topo = x.permute(0, 4, 1, 2, 3).contiguous()
        # d_topo: (n, m, c, t, v)
        d_topo = self.dynamic_topo_encode(d_topo)
        # d_topo: (n*m, d)

        x = x.permute(0, 4, 2, 3, 1).reshape(-1, V, C)
        x = self.channel_rise(x)
        x = x + self.point_pos_emb
        # x: (n*m*t, v, c)

        x = x.view(N, M, T, V, -1).permute(0, 1, 3, 4, 2).reshape(N, -1, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, -1, T).permute(0, 1, 3, 4, 2).reshape(N * M, -1, T, V)

        for layer in self.body:
            x = layer(x, d_topo)

        # x: (n*m, c, t, v)
        c_new = x.size(1)
        x = x.view(N, M, c_new, T, V)
        x = x.mean(-1).mean(1)
        x = self.drop_out(x)
        # x: (n, c, t)
        x = x.transpose(1, 2).contiguous()
        # x: (n, t, c)
        x = self.fc(x)
        return x


class BlockGCN_4_per_frame(nn.Module):
    def __init__(
        self,
        num_class,
        num_person,
        in_channels,
        graph_args,
        g_kernel_size=3,
        n_heads=8,
        t_kernel_size=5,
        data_bn=True,
        last_drop_out=0,
    ):
        super().__init__()
        self.graph = Graph(**graph_args)
        v = self.graph.num_node
        hop = self.graph.hop  # (v, v) If hop isn't symmetric, it should be transposed here.

        self.channel_rise = nn.Linear(in_channels, 128)
        self.point_pos_emb = nn.Parameter(torch.randn(1, v, 128))

        self.data_bn = nn.BatchNorm1d(num_person * v * 128) if data_bn else \
                       nn.Identity()

        self.dynamic_topo_encode = D_TopoEncoder(n_people=num_person)

        self.body = nn.ModuleList([
            GCN_MultiScale_TCN_Layer(128, 128, hop, g_kernel_size, n_heads, t_kernel_size),
            GCN_MultiScale_TCN_Layer(128, 256, hop, g_kernel_size, n_heads, t_kernel_size),
            GCN_MultiScale_TCN_Layer(256, 256, hop, g_kernel_size, n_heads, t_kernel_size),
            GCN_MultiScale_TCN_Layer(256, 256, hop, g_kernel_size, n_heads, t_kernel_size),
        ])
        
        self.drop_out = nn.Dropout(last_drop_out, inplace=True) if last_drop_out else \
                        nn.Identity()
        self.fc = nn.Linear(256, num_class)

    def forward(self, x: Tensor):
        N, C, T, V, M = x.size()

        d_topo = x.permute(0, 4, 1, 2, 3).contiguous()
        # d_topo: (n, m, c, t, v)
        d_topo = self.dynamic_topo_encode(d_topo)
        # d_topo: (n*m, d)

        x = x.permute(0, 4, 2, 3, 1).reshape(-1, V, C)
        x = self.channel_rise(x)
        x = x + self.point_pos_emb
        # x: (n*m*t, v, c)

        x = x.view(N, M, T, V, -1).permute(0, 1, 3, 4, 2).reshape(N, -1, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, -1, T).permute(0, 1, 3, 4, 2).reshape(N * M, -1, T, V)

        for layer in self.body:
            x = layer(x, d_topo)

        # x: (n*m, c, t, v)
        c_new = x.size(1)
        x = x.view(N, M, c_new, T, V)
        x = x.mean(-1).mean(1)
        x = self.drop_out(x)
        # x: (n, c, t)
        x = x.transpose(1, 2).contiguous()
        # x: (n, t, c)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    device = 'cpu'
    n, c, t, v, m = 5, 2, 30, 17, 2
    input_size = torch.Size([n, c, t, v, m])
    input_data = torch.randn(input_size).to(device)

    md = BlockGCN_6(
        num_class=34,
        in_channels=c,
        num_person=m,
        graph_args={
            'layout': 'coco',
            # There is no difference between setting
            # labeling_mode to 'uniform' and to 'spatial' here
            # because BlockGCN uses only the hop distance from the graph
            # to build the Static Topological Embedding matrix B.
        },
        g_kernel_size=1,
        n_heads=1,  # default 8 in BlockGCN
        # If 'n_heads' is bigger, learnable A becomes bigger, but W becomes smaller.
        # Choosing a suitable value depends on the min hidden channel size.
        t_kernel_size=9,
        data_bn=False,
        last_drop_out=0
    ).to(device)
    # output = md(input_data)
    # print(output.shape)
    summary(md, input_size, depth=3, device=device)
