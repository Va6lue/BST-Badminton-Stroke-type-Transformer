import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import v2
import numpy as np
from pathlib import Path


def get_merged_stroke_types(pad_to_same_len=False):
    class_ls = [
        '放小球', '擋小球', '殺球', '挑球',
        '長球', '平球', '切球', '推球',
        '撲球', '勾球', '發短球', '發長球'
    ]
    if pad_to_same_len:
        max_len = max([len(e) for e in class_ls])
        class_ls = [e.ljust(max_len, '　') for e in class_ls]
        class_ls = ['未知球種'.ljust(max_len, '　')+' '*7] + ['Top_'+s+' '*3 for s in class_ls] + ['Bottom_'+s for s in class_ls]
        return class_ls

    class_ls = ['未知球種'] + ['Top_'+s for s in class_ls] + ['Bottom_'+s for s in class_ls]
    return class_ls


def get_stroke_types(side='Both', pad_to_same_len=False):
    class_ls = [
        '放小球', '擋小球', '殺球', '點扣', '挑球', '防守回挑',
        '長球', '平球', '後場抽平球', '切球', '過渡切球', '推球',
        '撲球', '防守回抽', '勾球', '發短球', '發長球'
    ]
    if pad_to_same_len:
        max_len = max([len(e) for e in class_ls])
        class_ls = [e.ljust(max_len, '　') for e in class_ls]
        match side:
            case 'Both':
                class_ls = ['Top_'+s+' '*3 for s in class_ls] + ['Bottom_'+s for s in class_ls] + ['未知球種'.ljust(max_len, '　')+' '*7]
            case 'Top':
                class_ls = ['Top_'+s+' '*3 for s in class_ls] + ['未知球種'.ljust(max_len, '　')+' '*4]
            case 'Bottom':
                class_ls = ['Bottom_'+s for s in class_ls] + ['未知球種'.ljust(max_len, '　')+' '*7]

        return class_ls

    match side:
        case 'Both':
            class_ls = ['Top_'+s for s in class_ls] + ['Bottom_'+s for s in class_ls] + ['未知球種']
        case 'Top':
            class_ls = ['Top_'+s for s in class_ls] + ['未知球種']
        case 'Bottom':
            class_ls = ['Bottom_'+s for s in class_ls] + ['未知球種']
            
    return class_ls


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


class RandomTranslation(v2.Transform):
    '''Same as RandomTranslation in TemPose.'''
    def __init__(self, trans_range=(-0.3, 0.3), prob=0.3) -> None:
        super().__init__()
        self.trans_range = trans_range
        self.p = prob

    def __call__(self, x: np.ndarray):
        # x: (t, m, J, d)
        shift = np.random.uniform(*self.trans_range, size=x.shape[-1])
        if np.random.uniform(0, 1) < self.p:
            x = x + shift
        return x


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


class Dataset_npy(Dataset):
    def __init__(
        self,
        root_dir: Path,
        set_name: str,
        pose_style='J_only',
        seq_len=30
    ):
        super().__init__()
        assert set_name in ['train', 'val', 'test', 'test_specific'], 'Invalid set_name.'
        assert pose_style in ['J_only', 'JnB_interp', 'JnB_bone', 'Jn2B'], 'Invalid pose_style.'

        match set_name:
            case 'train':
                random_shift = RandomTranslation()
            case 'val' | 'test' | 'test_specific':
                random_shift = lambda x : x
        
        class_ls = get_stroke_types()

        # load .npy branch names
        data_branches = [str]
        labels = []

        if set_name != 'test_specific':
            target_dir = root_dir/set_name
            for typ in target_dir.iterdir():
                shots = sorted([str(s).replace('_pos.npy', '') for s in typ.glob('*_pos.npy')])
                data_branches += shots
                labels.append(np.full(len(shots), class_ls.index(typ.name), dtype=np.int64))
        else:
            data_branches = sorted([str(s).replace('_pos.npy', '') for s in root_dir.glob('*_pos.npy')])
            labels.append(np.full(len(data_branches), class_ls.index(root_dir.name), dtype=np.int64))

        self.data_branches = data_branches
        self.labels = np.concatenate(labels)

        self.pose_style = pose_style
        self.seq_len = seq_len
        self.random_shift = random_shift
        self.bone_pairs = get_bone_pairs(skeleton_format='coco')

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, i):
        joints = np.load(self.data_branches[i]+'_joints.npy')
        # joints: (t, m, J, d)
        pos = np.load(self.data_branches[i]+'_pos.npy')
        # pos: (t, m, xy)
        shuttle = np.load(self.data_branches[i]+'_shuttle.npy')
        # shuttle: (t, xy)
        
        joints: np.ndarray = joints.astype(np.float32)
        pos: np.ndarray = pos.astype(np.float32)
        shuttle: np.ndarray = shuttle.astype(np.float32)

        joints, pos, shuttle, new_video_len = make_seq_len_same(self.seq_len, joints, pos, shuttle)

        self.random_shift(joints)

        match self.pose_style:
            case 'J_only':
                human_pose = joints
            case 'JnB_interp':
                human_pose = interpolate_joints(joints, self.bone_pairs)
            case 'JnB_bone':
                bones = create_bones(joints, self.bone_pairs)
                human_pose = np.concatenate((joints, bones), axis=-2)
            case 'Jn2B':
                joints = interpolate_joints(joints, self.bone_pairs)
                bones = create_bones(joints, self.bone_pairs)
                human_pose = np.concatenate((joints, bones), axis=-2)
            case _:
                NotImplementedError

        # human_pose: (t, m, pose, d)
        # pos: (t, m, xy)
        # shuttle: (t, xy)
        # new_video_len: int
        # label: int
        return (human_pose, pos, shuttle), new_video_len, self.labels[i]


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


class Dataset_npy_collated_one_side(Dataset):
    def __init__(
        self,
        root_dir: Path,
        set_name: str,
        pose_style='J_only',
        use_top_side=True
    ):
        '''Use Top / Bottom labels only. Thus, the length of the dataset becomes half.

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
        self.labels = np.load(str(branch/'labels.npy'))

        unknown_i = len(get_stroke_types()) - 1
        n_single = unknown_i // 2
        if use_top_side:
            idx = (self.labels < n_single) | (self.labels == unknown_i)
            self.labels[self.labels == unknown_i] = n_single
        else:
            idx = self.labels >= n_single
            self.labels -= n_single
        
        self.human_pose = self.human_pose[idx]
        self.pos = self.pos[idx]
        self.shuttle = self.shuttle[idx]
        self.videos_len = self.videos_len[idx]
        self.labels = self.labels[idx]

        # J_only: (n, t, m, J, d)
        # JnB: (n, t, m, J+B, d)
        # Jn2B: (n, t, m, J+2B, d)
        # pos: (n, t, m, xy)
        # shuttle: (n, t, xy)
        # videos_len: (n)
        # labels: (n)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, i):
        return (self.human_pose[i], self.pos[i], self.shuttle[i]), \
                self.videos_len[i], self.labels[i]


class Dataset_npy_collated_single_pose(Dataset):
    def __init__(
        self,
        root_dir: Path,
        set_name: str,
        pose_style='J_only',
        opposite_on_purpose=False
    ):
        '''Use Top / Bottom pose only. The length of the dataset is unchanged.

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
        self.labels = np.load(str(branch/'labels.npy'))

        unknown_i = len(get_stroke_types()) - 1
        n_single = unknown_i // 2
        
        top_i = (self.labels < n_single)
        bot_i = ~top_i & (self.labels != unknown_i)
        idx = top_i | bot_i

        if opposite_on_purpose:
            top_i, bot_i = bot_i, top_i

        human_pose = np.empty_like(self.human_pose[:, :, 0:1, :, :])
        human_pose[top_i] = self.human_pose[top_i, :, 0:1, :, :]
        human_pose[bot_i] = self.human_pose[bot_i, :, 1:2, :, :]
        self.human_pose = human_pose[idx]

        self.pos = self.pos[idx]
        self.shuttle = self.shuttle[idx]
        self.videos_len = self.videos_len[idx]
        self.labels = self.labels[idx]

        # J_only: (n, t, m, J, d)
        # JnB: (n, t, m, J+B, d)
        # Jn2B: (n, t, m, J+2B, d)
        # pos: (n, t, m, xy)
        # shuttle: (n, t, xy)
        # videos_len: (n)
        # labels: (n)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, i):
        return (self.human_pose[i], self.pos[i], self.shuttle[i]), \
                self.videos_len[i], self.labels[i]


def prepare_npy_loaders(
    root_dir: Path,
    pose_style='Jn2B',
    seq_len=30,
    batch_size=128,
    use_cuda=True,
    num_workers=(0, 0, 0)
):
    train_set = Dataset_npy(root_dir, 'train', pose_style, seq_len)
    val_set = Dataset_npy(root_dir, 'val', pose_style, seq_len)
    test_set = Dataset_npy(root_dir, 'test', pose_style, seq_len)

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
        pin_memory=use_cuda,
        num_workers=num_workers[2]
    )
    return train_loader, val_loader, test_loader


def prepare_npy_collated_loaders(
    root_dir: Path,
    pose_style='Jn2B',
    batch_size=128,
    use_cuda=True,
    num_workers=(0, 0, 0),
    train_partial=1.0
):
    '''Notice that this one RandomTranslation is not used.'''
    train_set = Dataset_npy_collated(root_dir, 'train', pose_style, train_partial)
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


def prepare_npy_collated_one_side_loaders(
    root_dir: Path,
    pose_style='Jn2B',
    use_top_side=True,
    batch_size=128,
    use_cuda=True,
    num_workers=(0, 0, 0)
):
    '''Notice that this one RandomTranslation is not used.'''
    train_set = Dataset_npy_collated_one_side(root_dir, 'train', pose_style, use_top_side)
    val_set = Dataset_npy_collated_one_side(root_dir, 'val', pose_style, use_top_side)
    test_set = Dataset_npy_collated_one_side(root_dir, 'test', pose_style, use_top_side)

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


def prepare_npy_collated_single_pose_loaders(
    root_dir: Path,
    pose_style='Jn2B',
    opposite_on_purpose=False,
    batch_size=128,
    use_cuda=True,
    num_workers=(0, 0, 0)
):
    '''Notice that this one RandomTranslation is not used.'''
    train_set = Dataset_npy_collated_single_pose(root_dir, 'train', pose_style, opposite_on_purpose)
    val_set = Dataset_npy_collated_single_pose(root_dir, 'val', pose_style, opposite_on_purpose)
    test_set = Dataset_npy_collated_single_pose(root_dir, 'test', pose_style, opposite_on_purpose)

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


if __name__ == "__main__":
    # dataset = Dataset_npy(Path('dataset_npy'), 'train', seq_len=30)
    dataset = Dataset_npy_collated(Path('preparing_data/ShuttleSet_data/dataset_npy_collated'), 'test', train_partial=1)
    # dataset = Dataset_npy_collated_one_side(Path('dataset_npy_collated'), 'train')
    # dataset = Dataset_npy_collated_single_pose(Path('dataset_npy_collated'), 'train')
    print(len(dataset))
