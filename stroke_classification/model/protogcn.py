# Revealing Key Details to See Differences: A Novel Prototypical Perspective for Skeleton-based Action Recognition
# (2024/11) https://arxiv.org/abs/2411.18941
# Authors: Hongda Liu, Yunfan Liu, Min Ren, Hao Wang, Yunlong Wang, Zhenan Sun

# Modified by Jing-Yuan Chang

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchinfo import summary


class MTE_GCN(nn.Module):
    '''Motion Topology Enhancement GCN'''
    def __init__(self, in_channels, out_channels, A: Tensor):
        super().__init__()

        k = A.shape[0]  # K heads (simlar to ST-GCN's kernel size)
        mid_channels = out_channels // k
        
        ## for A = A0 + A_intra + A_inter
        self.A = nn.Parameter(A.clone())

        self.wq = nn.Conv2d(in_channels, mid_channels * k, 1)
        self.wk = nn.Conv2d(in_channels, mid_channels * k, 1)

        self.intra_act = nn.Softmax(-2)
        self.inter_act = nn.Tanh()
        
        self.intra_kernel = nn.Parameter(torch.zeros(k))
        self.inter_kernel = nn.Parameter(torch.zeros(k))
        ##

        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels * k, 1),
            nn.BatchNorm2d(mid_channels * k),
            nn.ReLU(inplace=True)
        )
        self.post = nn.Conv2d(mid_channels * k, out_channels, 1)

        self.bn = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.res = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.res = nn.Identity()

        self.act = nn.ReLU(inplace=True)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.k = k

    def forward(self, x: Tensor):
        n, c, t, v = x.shape
        res = self.res(x)

        A: Tensor = self.A.to(x.device)  # A: K V V
        A = A.view(1, self.k, 1, 1, v, v)
         
        hq: Tensor = self.wq(x).view(n, self.k, self.mid_channels, t, v)
        hw: Tensor = self.wk(x).view(n, self.k, self.mid_channels, t, v)
        
        # avg pooling on T
        hq = hq.mean(dim=-2, keepdim=True)
        hw = hw.mean(dim=-2, keepdim=True)
        # N K C' 1 V

        # build A_inter
        diff = hq.unsqueeze(-1) - hw.unsqueeze(-2)
        # N K C' 1 V V = N K C' 1 V 1 - N K C' 1 1 V
        A_inter = self.inter_act(diff)
        A_inter = A_inter * self.inter_kernel.view(1, self.k, 1, 1, 1, 1)
        A = A_inter + A
        # N K C' 1 V V = N K C' 1 V V + 1 K 1 1 V V

        # build A_intra
        A_intra = torch.einsum('nkctv,nkctw->nktvw', hq, hw).unsqueeze(2).contiguous()
        # N K 1 1 V V = einsum(N K C' 1 V * N K C' 1 V).unsqueeze(2)
        # N K 1 1 V V = (N K 1 V C' @ N K 1 C' V).unsqeuee(2)
        A_intra = self.intra_act(A_intra)
        A_intra = A_intra * self.intra_kernel.view(1, self.k, 1, 1, 1, 1)
        A = A_intra + A
        # N K C' 1 V V = N K 1 1 V V + N K C' 1 V V
        
        A = A.squeeze(3)
        # A: N K C' V V

        x = self.pre(x).view(n, self.k, self.mid_channels, t, v)
        x = x @ A
        # N K C' T V = N K C' T V @ N K C' V V
        x = x.view(n, -1, t, v)
        # N C T V
        x = self.post(x)
        x = self.act(self.bn(x) + res)
        # x: N C T V

        A_extra: Tensor = A_inter + A_intra
        # A_extra: N K C' 1 V V = N K C' 1 V V + N K 1 1 V V
        A_extra = A_extra.squeeze(3).view(n, -1, v*v)
        # A_extra: N C V*V
        return x, A_extra


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


class MultiScale_TCN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
        stride=1,
        num_joints=25,
        dropout=0.,
    ):
        '''
        ms_cfg: [(kernel_size, dilation), ..., ('max', kernel_size), '1x1']
        '''
        super().__init__()

        num_branches = len(ms_cfg)
        mid_channels = out_channels // num_branches
        rem_mid_channels = out_channels - mid_channels * (num_branches - 1)

        act = nn.ReLU(inplace=True)

        ## TCNs
        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == '1x1':
                branches.append(nn.Conv2d(in_channels, branch_c, kernel_size=1, stride=(stride, 1)))
                continue
            
            assert isinstance(cfg, tuple)
            if cfg[0] == 'max':
                branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, branch_c, kernel_size=1),
                    nn.BatchNorm2d(branch_c),
                    act,
                    nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0))
                ))
                continue
            
            assert isinstance(cfg[0], int) and isinstance(cfg[1], int)
            branches.append(nn.Sequential(
                nn.Conv2d(in_channels, branch_c, kernel_size=1),
                nn.BatchNorm2d(branch_c),
                act,
                TemporalConv2d(branch_c, branch_c, kernel_size=cfg[0], stride=stride, dilation=cfg[1], norm=False)
            ))

        self.branches = nn.ModuleList(branches)
        ##

        self.joints_coef = nn.Parameter(torch.zeros(num_joints))

        self.bottom = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            act,
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True)
        )

    def forward(self, x: Tensor):
        # x: N C T V
        x = torch.cat([x, x.mean(-1, keepdim=True)], dim=-1)
        
        branch_outs = []
        for t_conv in self.branches:
            out = t_conv(x)
            branch_outs.append(out)
        out = torch.cat(branch_outs, dim=1)

        local_feat = out[..., :-1].contiguous()
        global_feat = out[..., -1:].contiguous()
        feat = local_feat + global_feat * self.joints_coef
        # N C T V = N C T V + N C T 1 * V

        feat = self.bottom(feat)
        return feat


class GCN_TCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, n_joints=25, residual=True):
        super().__init__()

        self.gcn = MTE_GCN(in_channels, out_channels, A)
        self.tcn = MultiScale_TCN(out_channels, out_channels, stride=stride, num_joints=n_joints)
        self.act = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = nn.Identity()
        else:
            self.residual = TemporalConv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x: Tensor):
        res = self.residual(x)
        x, A_extra = self.gcn(x)
        x = self.tcn(x) + res
        return self.act(x), A_extra


class PRN(nn.Module):
    '''Prototype Reconstruction Network

    其實就是 activation 改成 softmax 然後不用 bias。
    '''
    def __init__(self, dim, n_prototype=100, dropout=0.1):
        super().__init__()
        self.q = nn.Linear(dim, n_prototype, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.assemble_memory = nn.Linear(n_prototype, dim, bias=False)
        self.drop = nn.Dropout(dropout, inplace=True)
        
    def forward(self, x):
        query = self.softmax(self.q(x))
        z = self.assemble_memory(query)
        return self.drop(z)


class ContrastiveLoss(nn.Module):
    '''Class-Specific Contrastive Loss'''
    def __init__(
        self,
        n_class,
        n_channel=625,
        d=256,
        mem_momentum=0.9,
        softmax_tmp=0.125,
        pred_threshold=0.0,
    ):
        super().__init__()

        self.register_buffer('hist_mem', torch.randn(d, n_class))

        self.lin = nn.Linear(n_channel, d)
        self.cross_entropy = nn.CrossEntropyLoss()

        self.n_class = n_class
        self.momentum = mem_momentum
        self.temperature = softmax_tmp
        self.pred_threshold = pred_threshold

    def get_tp_mask(self, pred_onehot, label_onehot, pred):
        tp = pred_onehot & label_onehot
        return tp & (pred > self.pred_threshold)

    def fetch_memory(self, f: Tensor, tp_mask: Tensor):
        # f: N d
        # tp_mask: N C
        f = f.transpose(0, 1).contiguous()
        # d N

        mask_sum = tp_mask.sum(0, keepdim=True)
        mask_sum[mask_sum == 0] = 1
        # 1 C <= N C

        # 依照 tp_mask 取出每個 class 成功判斷對應的 features 總和
        f_mask = f @ tp_mask.float()
        # d C = d N @ N C
        f_mask = f_mask / mask_sum
        
        hist_mem = self.hist_mem.to(f.device)
        mem = self.momentum * hist_mem + (1 - self.momentum) * f_mask

        if self.training:
            self.hist_mem = mem.detach()
        return mem  # d C

    def calculate_similarity(self, f: Tensor, mem: Tensor):
        # f: N d
        # mem: d C

        f = f / (torch.norm(f, dim=1, keepdim=True) + 1e-12)
        mem = mem / (torch.norm(mem, dim=0, keepdim=True) + 1e-12)

        sim = f @ mem  # N C = N d @ d C
        return sim / self.temperature
    
    def forward(self, logit: Tensor, z: Tensor, label: Tensor):
        # logit: N C
        # z: N V*V
        # label: N
        pred = logit.argmax(dim=1).long()  # N
        
        pred_onehot = F.one_hot(pred, self.n_class).bool()
        label_onehot = F.one_hot(label, self.n_class).bool()
        # N C

        prob = torch.softmax(logit, dim=1)  # N C

        f = self.lin(z)
        # f: N d
        tp_mask = self.get_tp_mask(pred_onehot, label_onehot, prob)
        # tp_mask: N C

        mem = self.fetch_memory(f, tp_mask)
        sim = self.calculate_similarity(f, mem)        
        # N C
        return self.cross_entropy(sim, label)


class ProtoGCN(nn.Module):
    def __init__(
        self,
        num_classes,
        n_nodes=25,
        in_channels=3,
        base_channels=96,
        num_stages=10,
        ch_ratio=2,
        inflate_stages=[5, 8],
        down_stages=[5, 8],
        num_prototype=100,
    ):
        super().__init__()
        
        A = self.random_graph(n_filters=8, n_nodes=n_nodes)

        ## GCN_TCN x L stages
        modules = [GCN_TCN_Block(in_channels, base_channels, A, n_joints=n_nodes, residual=False)]

        in_ch = base_channels
        out_ch = base_channels
        for i in range(2, num_stages + 1):
            stride = 2 if i in down_stages else 1
            if i in inflate_stages:
                out_ch *= ch_ratio
            modules.append(GCN_TCN_Block(in_ch, out_ch, A, stride, n_joints=n_nodes))
            in_ch = out_ch

        self.gcn_tcn_blocks = nn.ModuleList(modules)
        ##

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=2),  # N M C <= N M C 1 1
            nn.Linear(out_ch, num_classes)
        )

        # for Class-Specific Contrastive Learning
        self.prn = PRN(out_ch, num_prototype)
        self.post = nn.Sequential(
            nn.Flatten(start_dim=0, end_dim=-2),  # N*V*V C
            nn.Linear(out_ch, out_ch),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.out_ch = out_ch
    
    def random_graph(self, n_filters=8, n_nodes=25, std=.02, offset=.04):
        A = torch.randn(n_filters, n_nodes, n_nodes) * std + offset
        return A

    def forward(self, x: Tensor):
        N, M, C, T, V = x.size()
        x = x.view(N * M, C, T, V)

        for gcn_tcn in self.gcn_tcn_blocks:
            x, A_extra = gcn_tcn(x)
        
        x = x.view(N, M, self.out_ch, -1, V)
        x = self.classifier(x)
        x = x.mean(1)
        # x: N num_classes
        
        A_extra: Tensor = A_extra.view(N, M, -1, V * V).mean(1).view(N, -1, V * V)
        A_extra = A_extra.transpose(-2, -1).contiguous()
        # A_extra: N V*V C
        z: Tensor = self.prn(A_extra)
        z = self.post(z)  # N*V*V C
        z = z.mean(-1).view(N, V * V)
        return x, z


if __name__ == "__main__":
    N, M, C, T, V = 10, 2, 2, 30, 17
    n_class = 35
    device = 'cuda'

    model = ProtoGCN(
        num_classes=n_class,
        n_nodes=V,
        in_channels=C
    ).to(device)

    summary(model, input_size=(N, M, C, T, V), device=device)

    x, z = model(torch.randn(N, M, C, T, V).to(device))

    loss_fn = ContrastiveLoss(
        n_class=n_class,
        n_channel=V * V,
    ).to(device)

    label = torch.randint(0, n_class, (N,)).to(device)
    summary(loss_fn, input_data=[x, z, label], device=device)

    # loss = loss_fn(x, z, label)
    # print(loss)
