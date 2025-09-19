# ST-GCN: Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition
# (2018/01) https://arxiv.org/abs/1801.07455
# Authors: Sijie Yan, Yuanjun Xiong, Dahua Lin

# Modified by Jing-Yuan Chang

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchinfo import summary

import numpy as np


def get_hop_distance(num_node, edge, max_hop=1):
    '''Compute the hop count for all i to j.
    
    Note:
        `edge`: unidirectional
        Output: bidirectional
    '''
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    hop_dis = np.full((num_node, num_node), np.inf)
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A: Tensor):
    Dl = np.sum(A, 0)  # nodes in-degree
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = A @ Dn
    return AD


def normalize_undigraph(A: Tensor):
    Dl = np.sum(A, 0)  # nodes in-degree
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = Dn @ A @ Dn
    return DAD


class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """
    def __init__(self,
                 layout='coco',
                 strategy='saptial',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)

        self.hop_dis = get_hop_distance(self.num_node,
                                        self.edge,
                                        max_hop=max_hop)
        
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A  # this will be bidirectional.

    def get_edge(self, layout):
        match layout:
            case 'openpose':
                self.num_node = 18
                self_link = [(i, i) for i in range(self.num_node)]
                neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5),
                                (13, 12), (12, 11), (10, 9), (9, 8), (11, 5),
                                (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0),
                                (17, 15), (16, 14)]
                self.edge = self_link + neighbor_link
                self.center = 1
        
            case 'ntu-rgb+d':
                self.num_node = 25
                self_link = [(i, i) for i in range(self.num_node)]
                neighbor_1base = [(1, 2), (2, 21), (3, 21),
                                (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                                (10, 9), (11, 10), (12, 11), (13, 1), (14, 13),
                                (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                                (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
                neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
                self.edge = self_link + neighbor_link
                self.center = 21 - 1
        
            case 'ntu_edge':
                self.num_node = 24
                self_link = [(i, i) for i in range(self.num_node)]
                neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                                (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                                (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                                (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                                (23, 24), (24, 12)]
                neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
                self.edge = self_link + neighbor_link
                self.center = 2

            case 'coco':
                self.num_node = 17
                self_link = [(i, i) for i in range(self.num_node)]
                neighbor_1base = [
                    [16,14],[14,12],[17,15],[15,13],  # legs
                    [12,13],[6,12],[7,13],[6,7],      # torso
                    [8,6],[10,8],[9,7],[11,9],        # arms
                    [2,3],                            # between eyes
                    [2,1],[3,1],[4,2],[5,3],          # head
                    [4,6],[5,7]                       # ears to shoulders
                ]
                neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
                self.edge = self_link + neighbor_link
                self.center = 0

            case 'mediapipe':  # (x, y, z), dim=3
                self.num_node = 23
                self_link = [(i, i) for i in range(self.num_node)]
                neighbor_1base = [
                    (6,12),(6,10),(6,8),(8,10),       # left hand
                    (7,13),(7,11),(7,9),(9,11),      # right hand
                    (2,4),(4,6),(3,5),(5,7),          # arms
                    (2,3),(2,14),(3,15),(14,15),      # torso
                    (14,16),(15,17),(16,18),(17,19),  # legs
                    (18,20),(18,22),(20,22),          # left foot
                    (19,21),(19,23),(21,23)           # right foot
                ]
                neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
                self.edge = self_link + neighbor_link
                self.center = 0
        
            case _:
                raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_undigraph(adjacency)  # modified (from digraph to undigraph)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for j in range(self.num_node):      # target
                    for i in range(self.num_node):  # source (j's neighbors)
                        if self.hop_dis[i, j] == hop:  # i -> j
                            if self.hop_dis[self.center, i] == self.hop_dis[self.center, j]:
                                a_root[i, j] = normalize_adjacency[i, j]

                            elif self.hop_dis[self.center, i] < self.hop_dis[self.center, j]:
                                a_close[i, j] = normalize_adjacency[i, j]
                                # Node i is closer to center than Node j
                                # dist(center -> i) < dist(center -> j)
                            else:
                                a_further[i, j] = normalize_adjacency[i, j]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


class GCN_Block(nn.Module):
    """The basic module for applying a graph convolution.
    (Original name: ConvTemporalGraphical)

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels,
                              out_channels * kernel_size,
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1),
                              bias=bias)

    def forward(self, x: Tensor, A: Tensor):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)
        # Actually, it's a linear layer working on the channel dimension.
        # Each group has its own channel features.

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('n k c t v , k v w -> n c t w', x, A)
        # Get the sum of the features in the neighbor groups.

        return x.contiguous(), A


class ST_GCN_Block(nn.Module):
    """Applies a spatial temporal graph convolution over an input graph sequence.
    (Original name: st_gcn_block)

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1  # Temporal kernel size should be odd
        padding = ((kernel_size[0] - 1) // 2, 0)

        # Spatial kernel size is the number of the groups
        self.gcn = GCN_Block(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A


class ST_GCN_10(nn.Module):
    """Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_cfg (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)`
        
        where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """
    def __init__(self,
                 in_channels,
                 num_class,
                 graph_cfg,
                 edge_importance_weighting=True,
                 data_bn=True,
                 tem_kernel_size=9,
                 **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_cfg)
        A = torch.from_numpy(self.graph.A).type(torch.float)  # modified
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = tem_kernel_size
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1)) \
                       if data_bn else nn.Identity()
        
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList([
            ST_GCN_Block(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            ST_GCN_Block(64, 64, kernel_size, 1, **kwargs),
            ST_GCN_Block(64, 64, kernel_size, 1, **kwargs),
            ST_GCN_Block(64, 128, kernel_size, 1, **kwargs),
            ST_GCN_Block(128, 128, kernel_size, 1, **kwargs),
            ST_GCN_Block(128, 128, kernel_size, 1, **kwargs),
        ])

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(A.size()))
                for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(128, num_class, kernel_size=1)

    def forward(self, x: Tensor):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)
        return x


class ST_GCN_12(nn.Module):
    """Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_cfg (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)`
        
        where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """
    def __init__(self,
                 in_channels,
                 num_class,
                 graph_cfg,
                 edge_importance_weighting=True,
                 data_bn=True,
                 tem_kernel_size=9,
                 **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_cfg)
        A = torch.from_numpy(self.graph.A).type(torch.float)  # modified
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = tem_kernel_size
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1)) \
                       if data_bn else nn.Identity()
        
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList([
            ST_GCN_Block(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            ST_GCN_Block(64, 64, kernel_size, 1, **kwargs),
            ST_GCN_Block(64, 64, kernel_size, 1, **kwargs),
            ST_GCN_Block(64, 128, kernel_size, 1, **kwargs),
            ST_GCN_Block(128, 128, kernel_size, 1, **kwargs),
            ST_GCN_Block(128, 128, kernel_size, 2, **kwargs),
            ST_GCN_Block(128, 128, kernel_size, 1, **kwargs),
        ])

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(A.size()))
                for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(128, num_class, kernel_size=1)

    def forward(self, x: Tensor):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)
        return x


class ST_GCN_18(nn.Module):
    """Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_cfg (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)`
        
        where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """
    def __init__(self,
                 in_channels,
                 num_class,
                 graph_cfg,
                 edge_importance_weighting=True,
                 data_bn=True,
                 tem_kernel_size=9,
                 **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_cfg)
        A = torch.from_numpy(self.graph.A).type(torch.float)  # modified
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = tem_kernel_size
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1)) \
                       if data_bn else nn.Identity()
        
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList([
            ST_GCN_Block(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            ST_GCN_Block(64, 64, kernel_size, 1, **kwargs),
            ST_GCN_Block(64, 64, kernel_size, 1, **kwargs),
            ST_GCN_Block(64, 64, kernel_size, 1, **kwargs),
            ST_GCN_Block(64, 128, kernel_size, 2, **kwargs),
            ST_GCN_Block(128, 128, kernel_size, 1, **kwargs),
            ST_GCN_Block(128, 128, kernel_size, 1, **kwargs),
            ST_GCN_Block(128, 256, kernel_size, 2, **kwargs),
            ST_GCN_Block(256, 256, kernel_size, 1, **kwargs),
            ST_GCN_Block(256, 256, kernel_size, 1, **kwargs),
        ])

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(A.size()))
                for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x: Tensor):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)
        return x

    def extract_feature(self, x: Tensor):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)
        return output, feature


if __name__ == "__main__":
    md = ST_GCN_18(
        in_channels=2,
        num_class=35,
        graph_cfg={
            'layout': 'coco',
            'strategy': 'spatial'
        },
        edge_importance_weighting=True,
        data_bn=False,
        tem_kernel_size=9,
        dropout=0.5
    )

    n, c, t, v, m = 3, 2, 30, 17, 2
    input_size = torch.Size([n, c, t, v, m])
    summary(md, input_size, depth=3, device='cpu')
