import numpy as np
from pathlib import Path

from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor

from shuttleset_dataset import get_merged_stroke_types, get_bone_pairs, make_seq_len_same, create_bones, interpolate_joints


def pad_and_augment_one_npy_video(
    seq_len: int,
    joints: np.ndarray,
    pos: np.ndarray,
    shuttle: np.ndarray,
    bone_pairs: list[int, int]
):
    '''Pad to the same sequence length and Augment bones and interpolations.
    Input shape:
        `joints`: (t, 2, J, d)
        `pos`: (t, 2, xy)
        `shuttle`: (t, xy)
    output:
        J_only: (s, 2, J, d)
        JnB_interp: (s, 2, J+B, d)
        JnB_bone: (s, 2, J+B, d)
        Jn2B: (s, 2, J+2B, d)
        pos: (s, 2, xy)
        shuttle: (s, xy)
        video_len: int
    '''
    joints = joints.astype(np.float32)
    pos = pos.astype(np.float32)
    shuttle = shuttle.astype(np.float32)
    
    joints, pos, shuttle, new_video_len = make_seq_len_same(seq_len, joints, pos, shuttle)
    # assert len(shuttle) == seq_len, f'{seq_len}, {len(joints)}, {len(pos)}, {len(shuttle)}'

    joints_interpolated = interpolate_joints(joints, bone_pairs)
    bones = create_bones(joints, bone_pairs)

    JnB_bone = np.concatenate((joints, bones), axis=-2)
    Jn2B = np.concatenate((joints_interpolated, bones), axis=-2)
    
    return joints, joints_interpolated, JnB_bone, Jn2B, pos, shuttle, new_video_len


def collate_npy(root_dir: Path, set_name: str, seq_len: int, save_dir: Path):
    '''Collate .npy data before to make training faster.
    Notice: This will pad the arrays to the same length.
    '''
    assert set_name in ['train', 'val', 'test'], 'Invalid set_name.'
    
    class_ls = get_merged_stroke_types()

    # load .npy branch names
    data_branches = []
    labels = []
    target_dir = root_dir/set_name
    for typ in target_dir.iterdir():
        shots = sorted([str(s).replace('_pos.npy', '') for s in typ.glob('*_pos.npy')])
        data_branches += shots
        labels.append(np.full(len(shots), class_ls.index(typ.name), dtype=np.int64))
    labels = np.concatenate(labels)

    # load .npy files
    print(f'Load .npy files for {set_name} set ...')
    with ThreadPoolExecutor() as executor:
        tasks1: list[Future] = []
        tasks2: list[Future] = []
        tasks3: list[Future] = []

        for branch in data_branches:
            tasks1.append(executor.submit(np.load, branch+'_joints.npy'))
            tasks2.append(executor.submit(np.load, branch+'_pos.npy'))
            tasks3.append(executor.submit(np.load, branch+'_shuttle.npy'))

        joints_ls = [t1.result() for t1 in tasks1]
        pos_ls = [t2.result() for t2 in tasks2]
        shuttle_ls = [t3.result() for t3 in tasks3]
    print('Finish loading.')

    bone_pairs = get_bone_pairs(skeleton_format='coco')

    # Pad and Create bones and Interpolate
    print('Pad, Create bones and Interpolate ...')
    with ProcessPoolExecutor() as executor:
        tasks: list[Future] = []

        for joints, pos, shuttle in zip(joints_ls, pos_ls, shuttle_ls):
            tasks.append(executor.submit(
                pad_and_augment_one_npy_video,
                seq_len=seq_len,
                joints=joints,
                pos=pos,
                shuttle=shuttle,
                bone_pairs=bone_pairs
            ))

        J_ls = []
        JnB_interp_ls = []
        JnB_bone_ls = []
        Jn2B_ls = []
        pos_ls = []
        shuttle_ls = []
        videos_len = []

        for task in tasks:
            J_only, JnB_interp, JnB_bone, Jn2B, pos, shuttle, v_len = task.result()
            J_ls.append(J_only)
            JnB_interp_ls.append(JnB_interp)
            JnB_bone_ls.append(JnB_bone)
            Jn2B_ls.append(Jn2B)
            pos_ls.append(pos)
            shuttle_ls.append(shuttle)
            videos_len.append(v_len)
    
    J_only = np.stack(J_ls)
    JnB_interp = np.stack(JnB_interp_ls)
    JnB_bone = np.stack(JnB_bone_ls)
    Jn2B = np.stack(Jn2B_ls)
    pos = np.stack(pos_ls)
    shuttle = np.stack(shuttle_ls)
    videos_len = np.stack(videos_len)
    print('Finish padding and augmenting.')

    if not save_dir.is_dir():
        save_dir.mkdir()
    
    set_dir = save_dir/set_name
    if not set_dir.is_dir():
        set_dir.mkdir()

    np.save(str(set_dir/'J_only.npy'), J_only)
    np.save(str(set_dir/'JnB_interp.npy'), JnB_interp)
    np.save(str(set_dir/'JnB_bone.npy'), JnB_bone)
    np.save(str(set_dir/'Jn2B.npy'), Jn2B)
    np.save(str(set_dir/'pos.npy'), pos)
    np.save(str(set_dir/'shuttle.npy'), shuttle)
    np.save(str(set_dir/'videos_len.npy'), videos_len)
    np.save(str(set_dir/'labels.npy'), labels)
    print('Collation is complete.')


if __name__ == '__main__':
    seq_len = 30
    use_3d_pose = True

    preparing_root = Path('preparing_data/ShuttleSet_data_merged')

    str_3d = '_3d' if use_3d_pose else ''
    match seq_len:
        case 30:
            root_dir_raw = preparing_root/f'dataset{str_3d}_npy'
            save_root_dir_collate = preparing_root/f'dataset{str_3d}_npy_collated'
        case 100:
            root_dir_raw = preparing_root/f'dataset{str_3d}_npy_between_2_hits_with_max_limits'
            save_root_dir_collate = preparing_root/f'dataset{str_3d}_npy_collated_between_2_hits_with_max_limits_seq_100'
        case _:
            raise NotImplementedError(f'Invalid seq_len: {seq_len}. Must be 30 or 100.')

    ## Step 3 only
    # for set_name in ['train', 'val', 'test']:
    #     collate_npy(
    #         root_dir=root_dir_raw,
    #         set_name=set_name,
    #         seq_len=seq_len,
    #         save_dir=save_root_dir_collate
    #     )
