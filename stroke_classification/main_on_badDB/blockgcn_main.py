import torch
from torch import Tensor, nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_f1_score

import pandas as pd
from pathlib import Path
from copy import deepcopy
from collections import namedtuple
import time
from datetime import timedelta

import sys
import os
if __name__ == '__main__':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preparing_data.badmintonDB_dataset import prepare_npy_collated_loaders, \
                                               RandomTranslation_batch, \
                                               get_stroke_types
from model.blockgcn import BlockGCN_6, BlockGCN_10
from result_utils import show_f1_results, plot_confusion_matrix


Hyp = namedtuple('Hyp', [
    'n_epochs', 'batch_size', 'lr', 'weight_decay',
    'n_classes', 'seq_len', 'early_stop_n_epochs',
    'pose_style'
])
hyp = Hyp(
    n_epochs=1600,
    early_stop_n_epochs=300,
    batch_size=128,
    lr=5e-2,
    weight_decay=1e-2,
    n_classes=18,
    seq_len=72,
    pose_style='J_only'
)


def train_one_epoch(
    model: nn.Module,
    loader,
    random_shift_fn,
    loss_fn,
    optimizer: optim.Optimizer,
    device
):
    model.train()
    total_loss = 0.0

    for (human_pose, pos, shuttle), video_len, labels in loader:
        human_pose: Tensor = human_pose.to(device)
        labels: Tensor = labels.to(device)

        human_pose = random_shift_fn(human_pose)

        # human_pose: (n, t, 2, v, c)
        human_pose = human_pose.permute(0, 4, 1, 3, 2).contiguous()
        # human_pose: (n, c, t, v, m=2)
        logits = model(human_pose)
        loss: Tensor = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    train_loss = total_loss / len(loader)
    return train_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    loss_fn,
    loader,
    device
):
    model.eval()
    total_loss = 0.0
    cum_tp = torch.zeros(hyp.n_classes)
    cum_tn = torch.zeros(hyp.n_classes)
    cum_fp = torch.zeros(hyp.n_classes)
    cum_fn = torch.zeros(hyp.n_classes)

    for (human_pose, pos, shuttle), video_len, labels in loader:
        human_pose: Tensor = human_pose.to(device)
        labels: Tensor = labels.to(device)

        # human_pose: (n, t, 2, v, c)
        human_pose = human_pose.permute(0, 4, 1, 3, 2).contiguous()
        # human_pose: (n, c, t, v, m=2)
        logits = model(human_pose)
        loss: Tensor = loss_fn(logits, labels)
        total_loss += loss.item()

        pred = F.one_hot(torch.argmax(logits, dim=1), hyp.n_classes).bool()
        labels_onehot = F.one_hot(labels, hyp.n_classes).bool()

        tp = torch.sum(pred & labels_onehot, dim=0)
        tn = torch.sum(~pred & ~labels_onehot, dim=0)

        fp = torch.sum(pred & ~labels_onehot, dim=0)
        fn = torch.sum(~pred & labels_onehot, dim=0)

        cum_tp += tp.cpu()
        cum_tn += tn.cpu()
        cum_fp += fp.cpu()
        cum_fn += fn.cpu()

    val_loss = total_loss / len(loader)

    precision = cum_tp / (cum_tp + cum_fp)
    recall = cum_tp / (cum_tp + cum_fn)

    f1_score = 2 * precision * recall / (precision + recall)
    f1_score[f1_score.isnan()] = 0

    f1_score_avg = f1_score.mean()
    f1_score_min = f1_score.min()
    return val_loss, f1_score_avg, f1_score_min


@torch.no_grad()
def test(
    model: nn.Module,
    loader,
    device
):
    model.eval()
    pred_ls = []
    labels_ls = []
    for (human_pose, pos, shuttle), video_len, labels in loader:
        human_pose: Tensor = human_pose.to(device)

        # human_pose: (n, t, 2, v, c)
        human_pose = human_pose.permute(0, 4, 1, 3, 2).contiguous()
        # human_pose: (n, c, t, v, m=2)
        logits = model(human_pose)
        pred = torch.argmax(logits, dim=1).cpu()
        
        pred_ls.append(pred)
        labels_ls.append(labels)

    return torch.cat(pred_ls), torch.cat(labels_ls)


@torch.no_grad()
def test_topk(
    model: nn.Module,
    loader,
    device,
    k=2
):
    model.eval()
    pred_ls = []
    labels_ls = []
    for (human_pose, pos, shuttle), video_len, labels in loader:
        human_pose: Tensor = human_pose.to(device)

        # human_pose: (n, t, 2, v, c)
        human_pose = human_pose.permute(0, 4, 1, 3, 2).contiguous()
        # human_pose: (n, c, t, v, m=2)
        logits = model(human_pose)

        _, pred = torch.topk(logits, k=k, dim=1)
        
        pred_ls.append(pred.cpu())
        labels_ls.append(labels)

    return torch.cat(pred_ls), torch.cat(labels_ls)


def train_network(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    save_path: Path,
):
    random_shift_fn = RandomTranslation_batch()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=hyp.lr, weight_decay=hyp.weight_decay)

    best_value = 0.0
    early_stop_count = 0

    for epoch in range(1, hyp.n_epochs+1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            random_shift_fn=random_shift_fn,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        val_loss, f1_score_avg, f1_score_min = validate(
            model=model,
            loss_fn=loss_fn,
            loader=val_loader,
            device=device
        )
        t1 = time.time()
        print(f'Epoch({epoch}/{hyp.n_epochs}): train_loss={train_loss:.3f}, '\
              f'val_loss={val_loss:.3f}, macro_f1={f1_score_avg:.3f}, min_f1={f1_score_min:.3f} '\
              f'- {t1 - t0:.2f} s')

        early_stop_count += 1
        if best_value < f1_score_avg:
            best_value = f1_score_avg
            best_state = deepcopy(model.state_dict())
            print(f'Picked! => Best value {f1_score_avg:.3f}')
            early_stop_count = 0

        if early_stop_count == hyp.early_stop_n_epochs:
            print(f'Early stop with best value {best_value:.3f}')
            break
    
    torch.save(best_state, str(save_path))
    model.load_state_dict(best_state)
    return model


class Task:
    def __init__(self) -> None:
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'

    def prepare_dataloaders(
        self,
        root_dir: Path,
        pose_style='J_only',
    ):
        self.train_loader, \
        self.val_loader, \
        self.test_loader \
            = prepare_npy_collated_loaders(
                root_dir=root_dir,
                pose_style=pose_style,
                batch_size=hyp.batch_size,
                use_cuda=self.use_cuda,
                num_workers=(0, 0, 0),
            )
        self.pose_style = pose_style

    def get_network_architecture(self, model_name, in_channels=2):
        '''
        `model_name`
        - 'BlockGCN_6' (1.12M)
        - 'BlockGCN_10' (1.50M)
        '''
        match model_name:
            case 'BlockGCN_6':
                net = BlockGCN_6(
                    num_class=hyp.n_classes,
                    num_person=2,
                    in_channels=in_channels,
                    graph_args={
                        'layout': 'coco'
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
                )
            
            case 'BlockGCN_10':
                net = BlockGCN_10(
                    num_class=hyp.n_classes,
                    num_person=2,
                    in_channels=in_channels,
                    graph_args={
                        'layout': 'coco'
                        # There is no difference between setting
                        # labeling_mode to 'uniform' and to 'spatial' here
                        # because BlockGCN uses only the hop distance from the graph
                        # to build the Static Topological Embedding matrix B.
                    },
                    g_kernel_size=3,
                    n_heads=8,  # default 8 in BlockGCN
                    # If 'n_heads' is bigger, learnable A becomes bigger, but W becomes smaller.
                    # Choosing a suitable value depends on the min hidden channel size.
                    t_kernel_size=9,
                    data_bn=False,
                    last_drop_out=0
                )

            case _:
                raise NotImplementedError
        
        self.model_name = model_name
        self.net = net.to(self.device)

    def seek_network_weights(self, model_info='', serial_no=1):
        weight_exists = False

        model_info = f'_{model_info}' if model_info != '' else ''
        serial_str = f'_{serial_no}' if serial_no != 1 else ''
        
        model_postfix = model_info + serial_str

        save_name = self.model_name.lower() + model_postfix
        self.model_name += model_postfix

        weight_path = Path(f'weight_on_badDB/{save_name}.pt')
        if weight_path.exists():
            self.net.load_state_dict(torch.load(str(weight_path), map_location=self.device, weights_only=True))
            weight_exists = True
        else:
            train_t0 = time.time()
            self.net = train_network(
                model=self.net,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                device=self.device,
                save_path=weight_path
            )
            train_t1 = time.time()
            t = timedelta(seconds=int(train_t1 - train_t0))
            print(f'Total training time: {t}')
        
        return weight_exists

    def test(self, strokes_info_dir: Path, show_details=False, show_confusion_matrix=False):
        pred, gt = test(self.net, self.test_loader, self.device)
        print(f'Test (num_strokes: {len(pred)}) =>')

        f1_score_each = multiclass_f1_score(pred, gt, num_classes=hyp.n_classes, average=None)

        type_2_id = get_stroke_types(strokes_info_dir)
        class_ls = list(type_2_id.keys())[:(max(type_2_id.values()) + 1)]
        show_f1_results(
            model_name=self.model_name,
            f1_score_each=f1_score_each,
            class_ls=class_ls,
            show_details=show_details
        )

        acc = torch.sum(pred == gt).item() / len(pred)
        print('Accuracy:', f'{acc:.3f}')

        if show_confusion_matrix:
            plot_confusion_matrix(
                y_true=gt,
                y_pred=pred,
                need_pre_argmax=False,
                model_name=self.model_name,
                font_size=6,
                save=False
            )

    def test_topk_acc(self, k=2):
        assert k > 1, 'k should be > 1'
        pred, gt = test_topk(self.net, self.test_loader, self.device, k=k)
        gt = gt.unsqueeze(1).repeat(1, k)
        acc = torch.any(pred == gt, dim=1).sum().item() / len(gt)
        print(f'Top{k} Accuracy: {acc:.3f}')


if __name__ == '__main__':
    # Train and test on BadmintonDB data
    strokes_info_dir = Path('../BadmintonDB/after_generating')

    model_info = ''

    for serial_no in range(1, 6):
        print(f'Running serial {serial_no} ...')
        task = Task()
        task.prepare_dataloaders(
            root_dir=Path(f'preparing_data/BadmintonDB_data')\
                         /'dataset_npy_balance_collated',
            pose_style=hyp.pose_style,
        )
        task.get_network_architecture(model_name='BlockGCN_10', in_channels=2)
        weight_exists = task.seek_network_weights(model_info=model_info, serial_no=serial_no)
        task.test(strokes_info_dir, show_details=False, show_confusion_matrix=False)
        task.test_topk_acc(k=2)
        print('Serial', serial_no, 'done.')

        if not weight_exists:
            time.sleep(3)
