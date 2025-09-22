import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from pathlib import Path

import sys
import os
if __name__ == '__main__':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preparing_data.shuttleset_dataset import Dataset_npy_collated, \
                                              get_merged_stroke_types, get_stroke_types, get_bone_pairs
from model.bst import BST, BST_CG, BST_AP, BST_CG_AP


@torch.no_grad()
def infer(
    model: nn.Module,
    loader,
    device
):
    model.eval()
    pred_ls = []

    for (human_pose, pos, shuttle), video_len, labels in loader:
        human_pose: Tensor = human_pose.to(device)
        shuttle: Tensor = shuttle.to(device)
        pos: Tensor = pos.to(device)
        video_len: Tensor = video_len.to(device)

        human_pose = human_pose.view(*human_pose.shape[:-2], -1)
        logits = model(human_pose, shuttle, pos, video_len)

        pred = torch.argmax(logits, dim=1).cpu()
        
        pred_ls.append(pred)

    return torch.cat(pred_ls)


class Task:
    def __init__(self, n_joints=17) -> None:
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.n_joints = n_joints

    def prepare_loader(
        self,
        npy_collated_dir: Path,
        pose_style='Jn2B',
        batch_size=128,
    ):
        ####
        ## Replace 'your_set' here.
        ## Make sure your collated npy data has already normalized like I did.
        your_set = Dataset_npy_collated(npy_collated_dir, 'test', pose_style)
        ####

        self.infer_loader = DataLoader(
            dataset=your_set,
            batch_size=batch_size
        )
        self.pose_style = pose_style

    def get_network_architecture(
        self,
        model_name,
        seq_len=100,
        in_channels=2,
        n_classes=25,
    ):
        '''
        `model_name`
        - 'BST' (about 1.83M)
        - 'BST_CG' (about 1.85M)
        - 'BST_AP' (about 1.79M)
        - 'BST_CG_AP' (about 1.85M)
        '''
        n_bones = len(get_bone_pairs())

        match self.pose_style:
            case 'J_only':
                extra = 0
            case 'JnB_bone' | 'JnB_interp':
                extra = 1
            case 'Jn2B':
                extra = 2

        match model_name:
            case 'BST':
                net = BST(
                    in_dim=(self.n_joints + n_bones * extra) * in_channels,
                    n_class=n_classes,
                    seq_len=seq_len,
                    depth_tem=2,
                    depth_inter=1
                )

            case 'BST_CG':
                net = BST_CG(
                    in_dim=(self.n_joints + n_bones * extra) * in_channels,
                    n_class=n_classes,
                    seq_len=seq_len,
                    depth_tem=2,
                    depth_inter=1
                )

            case 'BST_AP':
                net = BST_AP(
                    in_dim=(self.n_joints + n_bones * extra) * in_channels,
                    n_class=n_classes,
                    seq_len=seq_len,
                    depth_tem=2,
                    depth_inter=1
                )

            case 'BST_CG_AP':
                net = BST_CG_AP(
                    in_dim=(self.n_joints + n_bones * extra) * in_channels,
                    n_class=n_classes,
                    seq_len=seq_len,
                    depth_tem=2,
                    depth_inter=1
                )

            case _:
                raise NotImplementedError
        
        self.net = net.to(self.device)

    def load_weight(self, weight_path: Path):
        self.net.load_state_dict(torch.load(str(weight_path), map_location=self.device, weights_only=True))

    def infer(self):
        return infer(self.net, self.infer_loader, self.device)


if __name__ == '__main__':
    # Infering example

    use_merged_ShuttleSet = True

    task = Task(n_joints=17)
    task.prepare_loader(
        npy_collated_dir=Path('preparing_data/ShuttleSet_data_merged')\
                        /"dataset_npy_collated_between_2_hits_with_max_limits_seq_100",
        pose_style="JnB_bone",  # 'J_only' or 'JnB_bone'
    )
    task.get_network_architecture(
        model_name='BST_CG_AP',
        seq_len=100,
        in_channels=2,
        n_classes=25 if use_merged_ShuttleSet else 35
    )
    task.load_weight(Path('weight')
                     /"bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_2.pt")
    
    pred = task.infer()
    # class IDs

    classes = get_merged_stroke_types() if use_merged_ShuttleSet else get_stroke_types()
    pred_cls = [classes[e] for e in pred]
    print(pred_cls)
