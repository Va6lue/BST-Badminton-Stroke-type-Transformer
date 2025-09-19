import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import v2
import pandas as pd
import numpy as np
from pathlib import Path


def get_stroke_types(strokes_info_dir: Path):
    # Get all merged types
    df = pd.read_csv(strokes_info_dir/'merged_strokes.csv')
    stroke_types_df = df[['merged_type', 'class_id']].drop_duplicates().reset_index(drop=True)
    type_2_id = stroke_types_df.set_index('merged_type')['class_id'].to_dict()
    
    # Set merged types
    merged_df = pd.read_csv(strokes_info_dir/'merged_stroke_types.csv')
    merged_type_2_type = merged_df.set_index('stroke_type')['become_stroke_type'].to_dict()
    for k, v in merged_type_2_type.items():
        type_2_id[k] = type_2_id[v]
    
    return type_2_id


def get_bone_pairs(skeleton_format='coco'):
    match skeleton_format:
        case 'coco':
            pairs = [
                (0,1),(0,2),(1,2),(1,3),(2,4),   # head
                (3,5),(4,6),                     # ears to shoulders
                (5,7),(7,9),(6,8),(8,10),        # arms
                (5,6),(5,11),(6,12),(11,12),     # torso
                (11,13),(13,15),(12,14),(14,16)  # legs
            ]
        case _:
            raise NotImplementedError
    return pairs


def make_seq_len_same(
    target_len: int,
    joints: np.ndarray,
    pos: np.ndarray,
    shuttle: np.ndarray
):
    video_len = len(pos)

    if video_len > target_len:
        need_padding = (video_len % target_len) > (target_len // 2)
        stride = video_len // target_len + int(need_padding)

        joints = joints[::stride][:target_len]
        pos = pos[::stride][:target_len]
        shuttle = shuttle[::stride][:target_len]

        new_video_len = len(pos)

        if need_padding:
            pad_len = target_len - new_video_len
            joints = np.pad(joints, ((0, pad_len), *([(0, 0)]*3)))
            pos = np.pad(pos, ((0, pad_len), *([(0, 0)]*2)))
            shuttle = np.pad(shuttle, ((0, pad_len), (0, 0)))

    else:
        # Since they have been normalized, we don't interpolate them.
        new_video_len = video_len

        pad_len = target_len - new_video_len
        joints = np.pad(joints, ((0, pad_len), *([(0, 0)]*3)))
        pos = np.pad(pos, ((0, pad_len), *([(0, 0)]*2)))
        shuttle = np.pad(shuttle, ((0, pad_len), (0, 0)))

    return joints, pos, shuttle, new_video_len


def create_bones(joints: np.ndarray, pairs) -> np.ndarray:
    '''Same as create_bones_robust in TemPose.'''
    # joints: (t, m, J, 2)
    bones = []
    for start, end in pairs:
        start_j = joints[:, :, start, :]
        end_j = joints[:, :, end, :]
        bone = np.where((start_j != 0.0) & (end_j != 0.0), end_j - start_j, 0.0)
        # bone: (t, m, 2)
        bones.append(bone)
    return np.stack(bones, axis=-2)


def interpolate_joints(joints: np.ndarray, pairs) -> np.ndarray:
    '''Same as create_limbs_robust when 'num_steps' set to 3 in TemPose.'''
    # joints: (t, m, J, 2)
    mid_joints = []
    for start, end in pairs:
        start_j = joints[:, :, start, :]
        end_j = joints[:, :, end, :]
        mid_j = np.where((start_j != 0.0) & (end_j != 0.0), (start_j + end_j) / 2, 0.0)
        # mid_j: (t, m, 2)
        mid_joints.append(mid_j)
    bones_center = np.stack(mid_joints, axis=-2)  # bones_center: (t, m, B, 2)
    return np.concatenate((joints, bones_center), axis=-2)  # (t, m, J+B, 2)


class RandomTranslation_batch(v2.Transform):
    '''Same as RandomTranslation in TemPose.'''
    def __init__(self, trans_range=(-0.3, 0.3), prob=0.3) -> None:
        super().__init__()
        self.trans_range = trans_range
        self.p = prob

    def __call__(self, x: Tensor):
        # x: (n, t, m, J, d)
        n = x.shape[0]
        d = x.shape[-1]
        shift = torch.from_numpy(
            np.random.uniform(*self.trans_range, size=(n, d)).astype(np.float32)
        ).to(x.device)
        if np.random.uniform(0, 1) < self.p:
            x = x + shift.view(n, 1, 1, 1, d)
        return x


class Dataset_npy_collated(Dataset):
    def __init__(
        self,
        root_dir: Path,
        set_name: str,
        pose_style='J_only',
        train_partial=1.0
    ):
        '''
        Parameters
        - `set_name`: 'train', 'val', 'test'
        - `pose_style`: 'J_only', 'JnB_interp', 'JnB_bone', 'Jn2B'
        
        Notice: There is no random translation here.
        '''
        super().__init__()
        
        assert set_name in ['train', 'val', 'test'], 'Invalid set_name.'
        assert pose_style in ['J_only', 'JnB_interp', 'JnB_bone', 'Jn2B'], 'Invalid pose_style.'

        branch = root_dir/set_name

        self.human_pose = np.load(str(branch/f'{pose_style}.npy'))
        self.pos = np.load(str(branch/'pos.npy'))
        self.shuttle = np.load(str(branch/'shuttle.npy'))
        self.videos_len = np.load(str(branch/'videos_len.npy'))
        self.labels: np.ndarray = np.load(str(branch/'labels.npy'))

        if set_name == 'train' and train_partial < 1:
            self.adjust_to_partial_train_set(train_partial)

        # J_only: (n, t, m, J, d)
        # JnB: (n, t, m, J+B, d)
        # Jn2B: (n, t, m, J+2B, d)
        # pos: (n, t, m, xy)
        # shuttle: (n, t, xy)
        # videos_len: (n)
        # labels: (n)

    def adjust_to_partial_train_set(self, train_partial):
        new_human_pose = []
        new_pos = []
        new_shuttle = []
        new_videos_len = []
        new_labels = []

        types = np.unique(self.labels)
        for typ in types:
            choose_i = np.nonzero(self.labels == typ)[0]
            typ_n = int(len(choose_i) * train_partial)
            choose_i = choose_i[:typ_n]

            new_human_pose.append(self.human_pose[choose_i])
            new_pos.append(self.pos[choose_i])
            new_shuttle.append(self.shuttle[choose_i])
            new_videos_len.append(self.videos_len[choose_i])
            new_labels.append(self.labels[choose_i])

        self.human_pose = np.concatenate(new_human_pose)
        self.pos = np.concatenate(new_pos)
        self.shuttle = np.concatenate(new_shuttle)
        self.videos_len = np.concatenate(new_videos_len)
        self.labels = np.concatenate(new_labels)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, i):
        return (self.human_pose[i], self.pos[i], self.shuttle[i]), \
                self.videos_len[i], self.labels[i]


def prepare_npy_collated_loaders(
    root_dir: Path,
    pose_style='Jn2B',
    batch_size=128,
    use_cuda=True,
    num_workers=(0, 0, 0),
):
    '''Notice that this one RandomTranslation is not used.'''
    train_set = Dataset_npy_collated(root_dir, 'train', pose_style)
    val_set = Dataset_npy_collated(root_dir, 'val', pose_style)
    test_set = Dataset_npy_collated(root_dir, 'test', pose_style)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_cuda,
        num_workers=num_workers[0]
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        pin_memory=use_cuda,
        num_workers=num_workers[1]
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        num_workers=num_workers[2]
    )
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    dataset_dir = Path('preparing_data/BadmintonDB_data/dataset_npy_balance_collated')
    dataset = Dataset_npy_collated(dataset_dir, set_name='train', pose_style='JnB_bone')
    print(len(dataset))
