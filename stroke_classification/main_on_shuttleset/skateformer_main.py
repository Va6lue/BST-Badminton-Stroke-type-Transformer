import torch
from torch import Tensor, nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_f1_score

import numpy as np
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

from preparing_data.shuttleset_dataset import prepare_npy_collated_loaders, \
                                              RandomTranslation_batch, Dataset_npy, \
                                              get_merged_stroke_types, get_stroke_types
from model.skateformer import SkateFormer
from result_utils import show_f1_results, plot_confusion_matrix


Hyp = namedtuple('Hyp', [
    'n_epochs', 'batch_size', 'lr',
    'n_classes', 'early_stop_n_epochs',
    'T', 'V', 'N', 'L',
    'seq_len', 'pose_style', 'use_3d_pose', 'train_partial'
])
hyp = Hyp(
    n_epochs=1600,
    early_stop_n_epochs=300,
    batch_size=128,
    lr=1e-3,
    n_classes=35,
    seq_len=30,
    T=32,
    N=4,
    V=25,
    L=5,
    pose_style='J_only',
    use_3d_pose=False,
    train_partial=1
)


def reset_joints_order_before_partitions(x: Tensor):
    # x: (..., V, C)
    head = np.arange(5)
    left_hand = np.array([9, 7, 5, 11, 6])
    right_hand = np.array([5, 6, 12, 8, 10])
    left_leg = np.array([5, 11, 12, 13, 15])
    right_leg = np.array([11, 6, 12, 14, 16])

    new_idx = torch.from_numpy(np.concatenate(
        [head, left_hand, right_hand, left_leg, right_leg]
    )).to(x.device)
    new_x = torch.index_select(x, dim=-2, index=new_idx)
    return new_x  # (..., V', C)


def pad_to_T(x: Tensor):
    # x: (B, S, m, V, C)
    S = x.shape[1]; T = hyp.T
    assert S <= T, 'T is too short.'
    pad_len = T - S
    new_x = F.pad(x, [0, 0, 0, 0, 0, 0, 0, pad_len])
    return new_x


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

        human_pose = reset_joints_order_before_partitions(human_pose)
        human_pose = pad_to_T(human_pose)

        # human_pose: (n, t, 2, v, c)
        human_pose = human_pose.permute(0, 4, 1, 3, 2).contiguous()
        # human_pose: (n, c, t, v, m=2)
        logits = model(human_pose, torch.arange(1, human_pose.shape[2]+1, device=human_pose.device))
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
        
        human_pose = reset_joints_order_before_partitions(human_pose)
        human_pose = pad_to_T(human_pose)

        # human_pose: (n, t, 2, v, c)
        human_pose = human_pose.permute(0, 4, 1, 3, 2).contiguous()
        # human_pose: (n, c, t, v, m=2)
        logits = model(human_pose, torch.arange(1, human_pose.shape[2]+1, device=human_pose.device))
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

        human_pose = reset_joints_order_before_partitions(human_pose)
        human_pose = pad_to_T(human_pose)

        # human_pose: (n, t, 2, v, c)
        human_pose = human_pose.permute(0, 4, 1, 3, 2).contiguous()
        # human_pose: (n, c, t, v, m=2)
        logits = model(human_pose, torch.arange(1, human_pose.shape[2]+1, device=human_pose.device))
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

        human_pose = reset_joints_order_before_partitions(human_pose)
        human_pose = pad_to_T(human_pose)

        # human_pose: (n, t, 2, v, c)
        human_pose = human_pose.permute(0, 4, 1, 3, 2).contiguous()
        # human_pose: (n, c, t, v, m=2)
        logits = model(human_pose, torch.arange(1, human_pose.shape[2]+1, device=human_pose.device))

        _, pred = torch.topk(logits, k=k, dim=1)
        
        pred_ls.append(pred.cpu())
        labels_ls.append(labels)

    return torch.cat(pred_ls), torch.cat(labels_ls)


def train_network(
    model: SkateFormer,
    train_loader,
    val_loader,
    device,
    save_path: Path,
):
    random_shift_fn = RandomTranslation_batch()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    para_group_dic = [
        {
            'params': [v for k, v in model.named_parameters() if k not in model.no_weight_decay()],
            'weight_decay': 5e-4  # original: 5e-4
        },
        {
            'params': [v for k, v in model.named_parameters() if k in model.no_weight_decay()],
            'weight_decay': 0.0
        }
    ]
    optimizer = optim.AdamW(para_group_dic, lr=hyp.lr)

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
    def __init__(self, shuttleset_merged=False) -> None:
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.shuttleset_merged = shuttleset_merged

    def prepare_dataloaders(
        self,
        root_dir: Path,
        pose_style='J_only',
        train_partial=1.0
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
                train_partial=train_partial
            )
        self.pose_style = pose_style

    def get_network_architecture(self, model_name, in_channels=2):
        '''
        `model_name`
        - 'SkateFormer_2' (2.38M)
        - 'SkateFormer_3' (3.52M)
        '''
        T = hyp.T; V = hyp.V  # (T frames, V joints) in total
        N = hyp.N; L = hyp.L  # (N frames, L joints) in first partition

        match model_name:
            case 'SkateFormer_2':
                assert T == 32 and N == 4, 'Please set T and N properly.'
                net = SkateFormer(
                    in_channels=in_channels,
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
                    num_people=2,
                    num_heads=32,
                    kernel_size=7,
                    type_1_size=(N, L),  # (N frames, L joints)
                    type_2_size=(N, V // L),  # (N frames, K distant joints)
                    type_3_size=(T // N, L),  # (M distant frames, L joints)
                    type_4_size=(T // N, V // L),  # (M distant frames, K distant joints)
                    use_index_t=True
                )

            case 'SkateFormer_3':
                assert T == 128 and N == 8, 'Please set T and N properly.'
                net = SkateFormer(
                    in_channels=in_channels,
                    channels=(96, 192, 192, 192),
                    depths=(2, 2, 2, 2),
                    # Down sampling D=(len(depths)-1) times, if stride is 2
                    # => T_final = T / 2^D
                    # => T_final still has to be divisible by N and M
                    # => You should pick T and N in the following constraints:
                    # <<< T = N*M and T_final % N == 0 and T_final % M == 0 >>>
                    # If 2^D = 2, min(T) = 4 = 2*2
                    # If 2^D = 4, min(T) = 16 = 4*4
                    # If 2^D = 8, min(T) = 64 = 8*8
                    # IF 2^D = 16, min(T) = 256 = 16*16
                    num_classes=hyp.n_classes,
                    num_frames=T,
                    num_points=V,
                    num_people=2,
                    num_heads=32,
                    kernel_size=7,
                    type_1_size=(N, L),  # (N frames, L joints)
                    type_2_size=(N, V // L),  # (N frames, K distant joints)
                    type_3_size=(T // N, L),  # (M distant frames, L joints)
                    type_4_size=(T // N, V // L),  # (M distant frames, K distant joints)
                    use_index_t=True
                )
            
            case _:
                raise NotImplementedError
        
        self.model_name = model_name
        self.net = net.to(self.device)

    def seek_network_weights(self, model_info='', serial_no=1):
        weight_exists = False

        model_info = f'_{model_info}' if model_info != '' else ''
        merged_info = '_merged' if self.shuttleset_merged else ''
        serial_str = f'_{serial_no}' if serial_no != 1 else ''

        model_postfix = model_info + merged_info + serial_str

        save_name = self.model_name.lower() + model_postfix
        self.model_name += model_postfix

        weight_path = Path(f'weight/{save_name}.pt')
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

    def test(self, show_details=False, show_confusion_matrix=False):
        pred, gt = test(self.net, self.test_loader, self.device)
        print(f'Test (num_strokes: {len(pred)}) =>')

        f1_score_each = multiclass_f1_score(pred, gt, num_classes=hyp.n_classes, average=None)
        show_f1_results(
            model_name=self.model_name,
            f1_score_each=f1_score_each,
            class_ls=get_merged_stroke_types(pad_to_same_len=True) if self.shuttleset_merged\
                     else get_stroke_types(pad_to_same_len=True),
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

    def compare_pred_gt_on_specific_type(self, dir_path: Path):
        infer_ds = Dataset_npy(
            root_dir=dir_path,
            set_name='test_specific',
            pose_style=self.pose_style,
            seq_len=hyp.T
        )
        infer_loader = DataLoader(
            dataset=infer_ds,
            batch_size=hyp.batch_size,
        )

        pred, gt = test(self.net, infer_loader, self.device)
        pred = pred.cpu().numpy()
        gt = gt.cpu().numpy()

        not_match = pred != gt
        class_ls = get_merged_stroke_types() if self.shuttleset_merged else get_stroke_types()
        with pd.option_context('display.max_rows', None):
            df = pd.DataFrame(
                data={
                    'Ball Round': [Path(e).stem for e in infer_ds.data_branches],
                    'Pred': [class_ls[e] if b else '-' for e, b in zip(pred, not_match)],
                    'GT': [class_ls[e] if b else '-' for e, b in zip(gt, not_match)]
                }
            )
            print(df)


if __name__ == '__main__':
    # Train and test on ShuttleSet data
    additional_model_info = 'some_no_weight_decay'

    match hyp.n_classes:
        case 25:
            merged = True
            merged_str = '_merged'
        case 35:
            merged = False
            merged_str = ''
        case _:
            raise NotImplementedError

    str_3d = '_3d' if hyp.use_3d_pose else ''
    match hyp.seq_len:
        case 30:
            npy_collated_dir = f'dataset{str_3d}_npy_collated'
            model_info = '3d' if hyp.use_3d_pose else ''
        case 100:
            npy_collated_dir = f'dataset{str_3d}_npy_collated_between_2_hits_with_max_limits_seq_100'
            model_info = f'between_2_hits_with_max_limits_seq_100{str_3d}'
        case _:
            raise NotImplementedError

    assert 0 < hyp.train_partial <= 1, 'hyp.train_partial should be in (0, 1].'
    if hyp.train_partial != 1:
        tmp_str = f'train_partial_0p{str(hyp.train_partial)[2:]}'
        if model_info != '':
            model_info += '_' + tmp_str
        else:
            model_info += tmp_str

    if additional_model_info != '':
        if model_info != '':
            model_info += f'_{additional_model_info}'
        else:
            model_info = additional_model_info

    for serial_no in range(1, 3):
        print(f'Running serial {serial_no} ...')
        task = Task(shuttleset_merged=merged)
        task.prepare_dataloaders(
            root_dir=Path(f'preparing_data/ShuttleSet_data{merged_str}')\
                            /npy_collated_dir,
            pose_style=hyp.pose_style,
            train_partial=hyp.train_partial
        )
        task.get_network_architecture(model_name='SkateFormer_2', in_channels=(3 if hyp.use_3d_pose else 2))
        weight_exists = task.seek_network_weights(model_info=model_info, serial_no=serial_no)
        task.test(show_details=False, show_confusion_matrix=False)
        task.test_topk_acc(k=2)
        # task.compare_pred_gt_on_specific_type(
        #     Path(f'preparing_data/ShuttleSet_data{merged_str}')
        #         /'dataset_npy/test/Top_點扣'
        # )
        print('Serial', serial_no, 'done.')

        if not weight_exists:
            time.sleep(3)
