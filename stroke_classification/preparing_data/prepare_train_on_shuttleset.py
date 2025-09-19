from mmpose.apis import MMPoseInferencer

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import subprocess
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor

from shuttleset_dataset import get_stroke_types, get_bone_pairs, make_seq_len_same, create_bones, interpolate_joints


def get_H(homography_info: pd.Series):
    '''Get from the pd object.'''
    h_str: str = homography_info['homography_matrix']
    H = h_str.strip().replace('[', '').replace(']', '').replace(',', '').split()
    H = np.array(list(map(float, H))).reshape((3, 3))
    return H


def get_corner_camera(homography_info: pd.Series):
    '''Get from the pd object.'''
    corner_camera = homography_info.loc['upleft_x':'downright_y']
    corner_camera = corner_camera.to_numpy(dtype=float).reshape((2, 4))
    return corner_camera


def scale_pos_by_resolution(arr: np.ndarray, width, height, aim_w=1280, aim_h=720):
    '''
    The shape of 2D `arr` is (2, N) or (3, N) if homogeneous.
    '''
    new_arr = arr.copy()
    new_arr[0, :] *= aim_w / width
    new_arr[1, :] *= aim_h / height
    return new_arr


def convert_homogeneous(arr: np.ndarray):
    '''
    The shape of 2D `arr` is (2, N). => The output will be (3, N).
    '''
    return np.concatenate((arr, np.full((1, arr.shape[-1]), 1.0)), axis=0)


def project(H: np.ndarray, P_prime: np.ndarray):
    '''
    Transform coordinates from the camera system to the court system.
    
    H: (3, 3)
    P_prime: (3, N)
    Output: (2, N)
    '''
    P = H @ P_prime
    P = P[:2, :] / P[-1, :]  # /= w
    return P


def get_court_info(homo_df: pd.DataFrame, vid: int):
    '''
    Get the homography matrix and the 4 corners of the court in the court coordinate corresponding to the video.
    '''
    homography_info = homo_df.loc[vid]

    H = get_H(homography_info)
    corner_camera = get_corner_camera(homography_info)
    corner_camera = convert_homogeneous(corner_camera)

    corner_court = project(H, corner_camera)
    return {
        'H': H,
        'border_L': corner_court[0, 0],
        'border_R': corner_court[0, 1],
        'border_U': corner_court[1, 0],
        'border_D': corner_court[1, 2],
    }


def to_court_coordinate(
    arr_camera: np.ndarray,
    vid: int,
    all_court_info: dict,
    res_df: pd.DataFrame
):
    '''
    Convert the camera coordinate system to the court coordinate system.

    If the camera coordinate is not from the resolution (1280, 720):
        It will be scaled to represent in (1280, 720).

    The shape of 2D `arr_camera` is (2, N).
    '''
    res_info = res_df.loc[vid]  # for resolution scaling
    H = all_court_info[vid]['H']

    arr_camera = scale_pos_by_resolution(arr_camera, width=res_info['width'], height=res_info['height'])
    arr_camera = convert_homogeneous(arr_camera)
    arr_court = project(H, arr_camera)
    return arr_court


def normalize_position(arr: np.ndarray, court_info: dict):
    '''
    Normalized by court boundary.

    `arr`: (2, N). There are N 'x' and N 'y'.
    Output: (2, N). Every 'x', 'y' in-court should be in [0, 1].
    '''
    x_dist = court_info['border_R'] - court_info['border_L']
    y_dist = court_info['border_D'] - court_info['border_U']

    x_normalized = (arr[0, :] - court_info['border_L']) / x_dist
    y_normalized = (arr[1, :] - court_info['border_U']) / y_dist
    return np.stack((x_normalized, y_normalized))


def normalize_joints(
    arr: np.ndarray,
    bbox: np.ndarray,
    v_height=None,
    center_align=False,
):
    '''
    - `arr`: (m, J, 2), m=2.
    - `bbox`: (m, 4), m=2.
    
    Output: (m, J, 2), m=2.
    '''
    # If v_height == None and center_align == False,
    # this normalization method is same as that used in TemPose.
    if v_height:
        dist = v_height / 4
    else:  # bbox diagonal dist
        dist = np.linalg.norm(bbox[:, 2:] - bbox[:, :2], axis=-1, keepdims=True)
    
    arr_x = arr[:, :, 0]
    arr_y = arr[:, :, 1]
    x_normalized = np.where(arr_x != 0.0, (arr_x - bbox[:, None, 0]) / dist, 0.0)
    y_normalized = np.where(arr_y != 0.0, (arr_y - bbox[:, None, 1]) / dist, 0.0)

    if center_align:
        center = (bbox[:, :2] + bbox[:, 2:]) / 2
        c_normalized = (center - bbox[:, :2]) / dist
        x_normalized -= c_normalized[:, None, 0]
        y_normalized -= c_normalized[:, None, 1]

    return np.stack((x_normalized, y_normalized), axis=-1)


def normalize_shuttlecock(arr: np.ndarray, v_width, v_height):
    '''
    Normalized by the video resolution.

    `arr`: (t, 2). There are t 'x' and t 'y'.
    Output: (t, 2). Every 'x', 'y' in-court should be in [0, 1].
    '''
    x_normalized = arr[:, 0] / v_width
    y_normalized = arr[:, 1] / v_height
    return np.stack((x_normalized, y_normalized), axis=-1)


def check_pos_in_court(keypoints: np.ndarray, vid: int, all_court_info: dict, res_df):
    '''
    The shape of `keypoints` is (m, J, 2).

    Output:
        in_court: (m)
        pos_court_normalized: (m, 2)
    '''
    n_people = keypoints.shape[0]
    
    feet_camera = keypoints[:, -2:, :]
    # feet_camera: (m, J, 2), J=2
    feet_camera = feet_camera.reshape(-1, 2).T
    # feet_camera: (2, m*J)

    feet_court = to_court_coordinate(feet_camera, vid=vid, all_court_info=all_court_info, res_df=res_df)
    feet_court = feet_court.reshape(2, n_people, -1)
    # feet_court: (2, m, J)

    pos_court = feet_court.mean(axis=-1)  # middle point between feet
    # pos_court: (2, m)
    pos_court_normalized = normalize_position(pos_court, court_info=all_court_info[vid]).T
    # pos_court_normalized: (m, 2)
    
    eps = 0.01  # soft border
    dim_in_court = (pos_court_normalized > -eps) & (pos_court_normalized < (1 + eps))
    in_court = dim_in_court[:, 0] & dim_in_court[:, 1]
    # in_court: (m)
    return in_court, pos_court_normalized


def detect_players_2d(
    inferencer: MMPoseInferencer,
    video_path: Path,
    all_court_info: dict,
    res_df: pd.DataFrame,
    J=17,
    normalized_by_v_height=False,
    center_align=False,
):
    '''
    Outputs
    -------
    failed_ls: list

    players_positions: (t, m, xy), m=xy=2
    
    players_joints: (t, m, J, xy), m=xy=2
    '''
    vid = int(video_path.name.split('_', 1)[0])

    failed_ls = []
    players_positions = []
    players_joints = []

    for frame_num, result in enumerate(inferencer(str(video_path), show=False)):
        keypoints = np.array([person['keypoints']
                              for person in result['predictions'][0]])  # batch_size=1 (default)
        # keypoints: (m, J, 2)

        # There should be at least 2 people in the video frame.
        failed = len(keypoints) < 2
        if not failed:
            in_court, pos_normalized = check_pos_in_court(keypoints, vid, all_court_info, res_df)
            # in_court: (m), pos_normalized: (m, xy), xy=2
            in_court_pid = np.nonzero(in_court)[0]
            
            # There should be 2 players only in a normal case.
            failed = len(in_court_pid) != 2
            if not failed:
                bboxes = np.array([person['bbox'][0]
                                   for person in result['predictions'][0]])  # batch_size=1 (default)
                # bboxes: (m, 4)

                # Make sure Top player before Bottom player (comparing y-dim)
                if pos_normalized[in_court_pid[0], 1] > pos_normalized[in_court_pid[1], 1]:
                    in_court_pid = np.flip(in_court_pid)
                
                failed_ls.append(False)
                players_positions.append(pos_normalized[in_court_pid])
                players_joints.append(normalize_joints(
                    arr=keypoints[in_court_pid],
                    bbox=bboxes[in_court_pid],
                    v_height=res_df.loc[vid, 'height'] if normalized_by_v_height else None,
                    center_align=center_align
                ))

        if failed:
            failed_ls.append(True)
            players_positions.append(np.zeros((2, 2), dtype=float))
            players_joints.append(np.zeros((2, J, 2), dtype=float))

    players_positions = np.stack(players_positions)
    # players_positions: (t, m, xy)
    players_joints = np.stack(players_joints)
    # players_joints: (t, m, J, xy)

    return failed_ls, players_positions, players_joints


def detect_players_3d(
    inferencer_2d: MMPoseInferencer,
    # inferencer_3d: MMPoseInferencer,
    video_path: Path,
    all_court_info: dict,
    res_df: pd.DataFrame,
    J=17,
):
    '''
    Outputs
    -------
    failed_ls: list

    players_positions: (t, m, xy), m=xy=2
    
    players_joints: (t, m, J, xy), m=xy=2
    '''
    vid = int(video_path.name.split('_', 1)[0])

    failed_ls = []
    players_positions = []
    players_joints = []

    gen_2d = inferencer_2d(str(video_path), show=False)
    inferencer_3d = MMPoseInferencer(pose3d='human3d')  # Should be written like this because there's a bug when you use it in '2d' way.
    gen_3d = inferencer_3d(str(video_path), show=False)
    
    for frame_num, (result_2d, result_3d) in enumerate(zip(gen_2d, gen_3d)):
        failed = False

        keypoints_2d = np.array([
            person['keypoints']
            for person in result_2d['predictions'][0]]  # batch_size=1 (default)
        )
        # keypoints_2d: (m, J, 2)

        keypoints_3d = np.array([
            person['keypoints']
            for person in result_3d['predictions'][0]]  # batch_size=1 (default)
        )
        # keypoints_3d: (m, J, 3)

        # There should be at least 2 people in the video frame.
        failed = len(keypoints_2d) < 2
        if not failed:
            in_court, pos_normalized = check_pos_in_court(keypoints_2d, vid, all_court_info, res_df)
            # in_court: (m), pos_normalized: (m, xy), xy=2
            in_court_pid = np.nonzero(in_court)[0]
            
            # There should be 2 players only in a normal case.
            failed = len(in_court_pid) != 2
            if not failed:
                # Make sure Top player before Bottom player (comparing y-dim)
                if pos_normalized[in_court_pid[0], 1] > pos_normalized[in_court_pid[1], 1]:
                    in_court_pid = np.flip(in_court_pid)
                
                failed_ls.append(False)
                players_positions.append(pos_normalized[in_court_pid])
                players_joints.append(keypoints_3d[in_court_pid])

        if failed:
            failed_ls.append(True)
            players_positions.append(np.zeros((2, 2), dtype=float))
            players_joints.append(np.zeros((2, J, 3), dtype=float))

    players_positions = np.stack(players_positions)
    # players_positions: (t, m, xy)
    players_joints = np.stack(players_joints)
    # players_joints: (t, m, J, xyz)

    return failed_ls, players_positions, players_joints


def detect_shuttlecock_by_TrackNetV3_with_attension(
    cur_i: int,
    total_tasks: int,
    video_path: Path,
    save_dir: Path,
    model_folder=Path("C:/MyResearch/TrackNetV3-main"),
):
    '''
    TrackNetV3 (using attention)
    https://github.com/alenzenx/TrackNetV3
    '''
    process_args = [
        'python', str(model_folder/'predict.py').replace('\\', '/'),
        "--video_file", str(video_path).replace('\\', '/'),
        "--model_file", str(model_folder/"exp"/"model_best.pt").replace('\\', '/'),
        "--save_dir", str(save_dir).replace('\\', '/'),
        # "--output_video"  # added myself
        # "--verbose"  # added myself
    ]
    r = subprocess.run(process_args)
    assert r.returncode == 0, 'Subprocess failed!'

    type_path = video_path.parent
    set_name = type_path.parent.name
    print(f'Shuttlecock detection ({cur_i}/{total_tasks}): {set_name}/{type_path.name}/{video_path.name} done!')


def detect_shuttlecock_by_TrackNetV3_with_rectification(
    cur_i: int,
    total_tasks: int,
    video_path: Path,
    save_dir: Path,
    model_folder=Path("C:/MyResearch/TrackNetV3-master"),
):
    '''
    TrackNetV3 (with rectification module)
    https://github.com/qaz812345/TrackNetV3
    '''
    process_args = [
        "python", str(model_folder/'predict.py').replace('\\', '/'),
        "--video_file", str(video_path).replace('\\', '/'),
        "--tracknet_file", str(model_folder/'weight'/'TrackNet_best.pt').replace('\\', '/'),
        "--inpaintnet_file", str(model_folder/'weight'/'InpaintNet_best.pt').replace('\\', '/'),
        "--save_dir", str(save_dir).replace('\\', '/'),
        # "--output_video",
        "--large_video"
    ]
    subprocess.run(process_args)
    r = subprocess.run(process_args)
    assert r.returncode == 0, 'Subprocess failed!'

    type_path = video_path.parent
    set_name = type_path.parent.name
    print(f'Shuttlecock detection ({cur_i}/{total_tasks}): {set_name}/{type_path.name}/{video_path.name} done!')


def get_shuttle_result(path: Path, v_width, v_height):
    df = pd.read_csv(str(path)).drop_duplicates('Frame')  # for the .csv generated by TrackNetV3 with attension
    df = df.set_index('Frame').drop(columns='Visibility')
    shuttle_camera = df.to_numpy().astype(float)
    # shuttle_camera: (t, 2)
    return normalize_shuttlecock(shuttle_camera, v_width, v_height)


def mk_same_dir_structure(src_dir: Path, target_dir: Path, root=True):
    '''The roots can be different. Other subdirectories should be all the same.'''
    if root and not target_dir.is_dir():
        target_dir.mkdir()
    for src_sub_dir in src_dir.iterdir():
        if src_sub_dir.is_dir():
            target_sub_dir = target_dir/src_sub_dir.name
            if not target_sub_dir.is_dir():
                target_sub_dir.mkdir()
            mk_same_dir_structure(src_sub_dir, target_sub_dir, root=False)


def prepare_trajectory(
    my_clips_folder: Path,
    model_folder: Path,
    save_shuttle_dir: Path,
):
    '''Trajectory detection

    Notice: max_workers shouldn't be too high because this process is using GPU as well.
    '''
    all_mp4_paths = sorted(my_clips_folder.glob('**/*.mp4'))

    with ProcessPoolExecutor(max_workers=4) as executor:
        for i, video_path in enumerate(all_mp4_paths, start=1):
            shuttle_result_path = save_shuttle_dir/(video_path.stem+'_ball.csv')
            if not shuttle_result_path.exists():
                executor.submit(
                    detect_shuttlecock_by_TrackNetV3_with_attension,
                    i, len(all_mp4_paths),
                    video_path=video_path,
                    save_dir=save_shuttle_dir,
                    model_folder=model_folder
                )


def prepare_2d_dataset_npy_from_raw_video(
    my_clips_folder: Path,
    save_shuttle_dir: Path,
    save_root_dir: Path,
    resolution_df: pd.DataFrame,
    all_court_info: dict,
    joints_normalized_by_v_height=False,
    joints_center_align=False
):
    # Make sure there are folders that can contain .npy files.
    mk_same_dir_structure(src_dir=my_clips_folder, target_dir=save_root_dir)

    all_mp4_paths = sorted(my_clips_folder.glob('**/*.mp4'))

    pose_inferencer = MMPoseInferencer('human')

    pbar = tqdm(range(len(all_mp4_paths)), desc='Yield .npy files', unit='video')
    for video_path in all_mp4_paths:
        # Set the save paths.
        ball_type_dir = video_path.parent
        set_split_dir = ball_type_dir.parent
        save_branch = str(save_root_dir/set_split_dir.name/ball_type_dir.name/video_path.stem)

        if not Path(save_branch+'_shuttle.npy').exists():
            # Players detection
            failed_ls, players_positions, joints = \
                detect_players_2d(
                    inferencer=pose_inferencer,
                    video_path=video_path,
                    all_court_info=all_court_info,
                    res_df=resolution_df,
                    normalized_by_v_height=joints_normalized_by_v_height,
                    center_align=joints_center_align
                )
            
            # Get the shuttlecock position from the generated .csv files.
            shuttle_result_path = save_shuttle_dir/(video_path.stem+'_ball.csv')            
            vid = int(video_path.name.split('_', 1)[0])
            shuttle_result = get_shuttle_result(
                path=shuttle_result_path,
                v_width=resolution_df.loc[vid, 'width'],
                v_height=resolution_df.loc[vid, 'height']
            )
            # shuttle_result: (F, 2)

            # Set the content of the frame failed in players detection to 0
            if np.any(failed_ls):
                shuttle_result[failed_ls, :] = 0

            np.save(save_branch+'_pos.npy', players_positions)
            # (F, P, xy)
            np.save(save_branch+'_joints.npy', joints)
            # (F, P, J, xy)
            np.save(save_branch+'_shuttle.npy', shuttle_result)
            # (F, xy)
        pbar.update()
    pbar.close()


def prepare_3d_dataset_npy_from_raw_video(
    my_clips_folder: Path,
    save_shuttle_dir: Path,
    save_root_dir: Path,
    resolution_df: pd.DataFrame,
    all_court_info: dict,
):
    # Make sure there are folders that can contain .npy files.
    mk_same_dir_structure(src_dir=my_clips_folder, target_dir=save_root_dir)

    all_mp4_paths = sorted(my_clips_folder.glob('**/*.mp4'))

    pose_inferencer_2d = MMPoseInferencer('human')
    # pose_inferencer_3d = MMPoseInferencer(pose3d='human3d')

    pbar = tqdm(range(len(all_mp4_paths)), desc='Yield .npy files', unit='video')
    for video_path in all_mp4_paths:
        # Set the save paths.
        ball_type_dir = video_path.parent
        set_split_dir = ball_type_dir.parent
        save_branch = str(save_root_dir/set_split_dir.name/ball_type_dir.name/video_path.stem)

        if not Path(save_branch+'_shuttle.npy').exists():
            # Players detection
            failed_ls, players_positions, joints = \
                detect_players_3d(
                    inferencer_2d=pose_inferencer_2d,
                    # inferencer_3d=pose_inferencer_3d,
                    video_path=video_path,
                    all_court_info=all_court_info,
                    res_df=resolution_df,
                )
            
            # Get the shuttlecock position from the generated .csv files.
            shuttle_result_path = save_shuttle_dir/(video_path.stem+'_ball.csv')            
            vid = int(video_path.name.split('_', 1)[0])
            shuttle_result = get_shuttle_result(
                path=shuttle_result_path,
                v_width=resolution_df.loc[vid, 'width'],
                v_height=resolution_df.loc[vid, 'height']
            )
            # shuttle_result: (F, 2)

            # Set the content of the frame failed in players detection to 0
            if np.any(failed_ls):
                shuttle_result[failed_ls, :] = 0

            np.save(save_branch+'_pos.npy', players_positions)
            # (F, P, xy)
            np.save(save_branch+'_joints.npy', joints)
            # (F, P, J, xyz)
            np.save(save_branch+'_shuttle.npy', shuttle_result)
            # (F, xy)
        pbar.update()
    pbar.close()


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
    
    class_ls = get_stroke_types()

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
    use_3d_pose = False

    my_clips_root = Path('C:/BST_advanced/ShuttleSet')
    preparing_root = Path('preparing_data/ShuttleSet_data')

    str_3d = '_3d' if use_3d_pose else ''
    match seq_len:
        case 30:
            my_clips_folder = my_clips_root/"shuttle_set"
            
            # Save directories (Step 1-3)
            save_shuttle_dir = preparing_root/'shuttlecock_temp'
            save_root_dir_raw = preparing_root/f'dataset{str_3d}_npy'
            save_root_dir_collate = preparing_root/f'dataset{str_3d}_npy_collated'
        
        case 100:
            my_clips_folder = my_clips_root/"shuttle_set_between_2_hits_with_max_limits"
            
            # Save directories (Step 1-3)
            save_shuttle_dir = preparing_root/'shuttlecock_temp_between_2_hits_with_max_limits'
            save_root_dir_raw = preparing_root/f'dataset{str_3d}_npy_between_2_hits_with_max_limits'
            save_root_dir_collate = preparing_root/f'dataset{str_3d}_npy_collated_between_2_hits_with_max_limits_seq_100'

        case _:
            raise NotImplementedError(f'Invalid seq_len: {seq_len}. Must be 30 or 100.')

    homo_df = pd.read_csv(my_clips_root/"set/homography.csv").set_index('id')
    resolution_df = pd.read_csv(my_clips_root/"my_raw_video_resolution.csv").set_index('id')

    all_court_info = {vid: get_court_info(homo_df, vid)
                      for vid in resolution_df.index}

    ## I recommended to run each Step individually.

    ## Step 1
    # prepare_trajectory(
    #     my_clips_folder=my_clips_folder,
    #     model_folder=Path("C:/MyResearch/TrackNetV3-main"),
    #     save_shuttle_dir=save_shuttle_dir
    # )

    ## Step 2 (choose 2d or 3d)
    # prepare_2d_dataset_npy_from_raw_video(
    #     my_clips_folder,
    #     save_shuttle_dir,
    #     save_root_dir_raw,
    #     resolution_df,
    #     all_court_info,
    #     joints_normalized_by_v_height=False,
    #     joints_center_align=True
    # )
    # prepare_3d_dataset_npy_from_raw_video(
    #     my_clips_folder,
    #     save_shuttle_dir,
    #     save_root_dir_raw,
    #     resolution_df,
    #     all_court_info,
    # )

    ## Step 3
    # for set_name in ['train', 'val', 'test']:
    #     collate_npy(
    #         root_dir=save_root_dir_raw,
    #         set_name=set_name,
    #         seq_len=seq_len,
    #         save_dir=save_root_dir_collate
    #     )
