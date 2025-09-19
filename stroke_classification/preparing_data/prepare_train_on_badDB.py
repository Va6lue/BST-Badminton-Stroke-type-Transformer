from mmpose.apis import MMPoseInferencer

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import subprocess
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor

from badmintonDB_dataset import get_stroke_types, get_bone_pairs, make_seq_len_same, create_bones, interpolate_joints


def get_corners_camera_all_videos(court_detection_dir: Path):
    all_corners_camera = dict()
    match_court_paths = sorted(court_detection_dir.glob('*.csv'))
    for court_path in match_court_paths:
        m = int(court_path.stem[5])
        df = pd.read_csv(court_path, header=None, sep=';')
        corner_camera = df.to_numpy(dtype=np.float64)[:-2]  # (4, 2)
        all_corners_camera[m] = corner_camera
    return all_corners_camera


def get_H_all_videos(all_corners_camera: dict):
    corners_court = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    return {k: cv2.findHomography(arr, corners_court)[0] for k, arr in all_corners_camera.items()}


def scale_pos_by_resolution(arr: np.ndarray, width, height, aim_w=640, aim_h=360):
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


def to_court_coordinate(
    arr_camera: np.ndarray,
    vid: int,
    all_H: dict,
    res_df: pd.DataFrame = None,
):
    '''
    Convert the camera coordinate system to the court coordinate system.

    If the camera coordinate is not from the resolution (640, 360):
        It will be scaled to represent in (640, 360).

    The shape of 2D `arr_camera` is (2, N).
    '''
    if res_df is not None:
        res_info = res_df.loc[vid]  # for resolution scaling
        arr_camera = scale_pos_by_resolution(arr_camera, width=res_info['width'], height=res_info['height'])
    arr_camera = convert_homogeneous(arr_camera)
    arr_court = project(all_H[vid], arr_camera)
    return arr_court


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


def check_pos_in_court(
    keypoints: np.ndarray,
    vid: int,
    res_df: pd.DataFrame,
    all_H: pd.DataFrame,
):
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

    feet_court = to_court_coordinate(feet_camera, vid=vid, res_df=res_df, all_H=all_H)
    feet_court = feet_court.reshape(2, n_people, -1)
    # feet_court: (2, m, J)

    pos_court_normalized = feet_court.mean(axis=-1).T  # middle point between feet
    # pos_court_normalized: (m, 2)
    
    eps = 0.01  # soft border
    dim_in_court = (pos_court_normalized > -eps) & (pos_court_normalized < (1 + eps))
    in_court = dim_in_court[:, 0] & dim_in_court[:, 1]
    # in_court: (m)
    return in_court, pos_court_normalized


def detect_players_2d(
    inferencer: MMPoseInferencer,
    vid: int,
    video_path: Path,
    res_df: pd.DataFrame,
    all_H: pd.DataFrame,
    J=17,
    normalized_by_v_height=False,
    center_align=False,
):
    '''
    Outputs
    -------
    players_positions: (t, m, xy), m=xy=2
    
    players_joints: (t, m, J, xy), m=xy=2
    '''
    players_positions = []
    players_joints = []

    for frame_num, result in enumerate(inferencer(str(video_path), show=False)):
        keypoints = np.array([person['keypoints']
                              for person in result['predictions'][0]])  # batch_size=1 (default)
        # keypoints: (m, J, 2)

        # There should be at least 2 people in the video frame.
        failed = len(keypoints) < 2
        if not failed:
            in_court, pos_normalized = check_pos_in_court(keypoints, vid, res_df=res_df, all_H=all_H)
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
                
                players_positions.append(pos_normalized[in_court_pid])
                players_joints.append(normalize_joints(
                    arr=keypoints[in_court_pid],
                    bbox=bboxes[in_court_pid],
                    v_height=res_df.loc[vid, 'height'] if normalized_by_v_height else None,
                    center_align=center_align
                ))

        if failed:
            players_positions.append(np.zeros((2, 2), dtype=float))
            players_joints.append(np.zeros((2, J, 2), dtype=float))

    players_positions = np.stack(players_positions)
    # players_positions: (t, m, xy)
    players_joints = np.stack(players_joints)
    # players_joints: (t, m, J, xy)

    return players_positions, players_joints


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
    print(f'Shuttlecock detection ({cur_i}/{total_tasks}): {type_path.name}/{video_path.name} done!')


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


def check_empty_file_func(p: Path):
    if p.stat().st_size == 0:
        return p, True
    return p, False


def check_empty_files_and_del(root: Path):
    all_csv_files = list(root.glob('*.csv'))

    with ThreadPoolExecutor() as executor:
        tasks = [executor.submit(check_empty_file_func, p) for p in all_csv_files]
        results = [task.result() for task in tasks]
    
    empty_files = [p for p, is_empty in results if is_empty]
    for p in empty_files:
        p.unlink()
    return empty_files


def del_csv_files(shuttle_unmatch_ls: list[str], root: Path):
    csv_files = [root/f'{stem}_ball.csv' for stem in shuttle_unmatch_ls]
    for p in csv_files:
        p.unlink(missing_ok=True)


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


def split_dataset(badDB_root: Path):
    strokes_info_path = badDB_root/'after_generating'/'merged_strokes.csv'
    df = pd.read_csv(strokes_info_path)

    df['set'] = None
    for name, group in df.groupby('merged_type'):
        n = len(group)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        
        train_index = group.index[:n_train]
        val_index = group.index[n_train:n_train + n_val]
        test_index = group.index[n_train + n_val:]
        
        df.loc[train_index, 'set'] = 'train'
        df.loc[val_index, 'set'] = 'val'
        df.loc[test_index, 'set'] = 'test'
    
    train_df = df.loc[df['set'] == 'train']
    val_df = df.loc[df['set'] == 'val']
    test_df = df.loc[df['set'] == 'test']

    train_cnt = train_df.groupby('merged_type').size()
    val_cnt = val_df.groupby('merged_type').size()
    test_cnt = test_df.groupby('merged_type').size()

    cnt_df = pd.DataFrame({
        'Train': train_cnt,
        'Val': val_cnt,
        'Test': test_cnt,
        'Total': train_cnt + val_cnt + test_cnt
    })
    cnt_df.loc['Sum'] = cnt_df.sum()

    return df, cnt_df


def prepare_2d_dataset_npy_from_raw_video(
    my_clips_folder: Path,
    save_shuttle_dir: Path,
    save_root_dir: Path,
    strokes_df: pd.DataFrame,
    resolution_df: pd.DataFrame,
    all_H: pd.DataFrame,
    joints_normalized_by_v_height=False,
    joints_center_align=True
):
    shuttle_unmatch_ls = []

    if not save_root_dir.is_dir():
        save_root_dir.mkdir()

    # Make sure there are folders that can contain .npy files.
    mk_same_dir_structure(src_dir=my_clips_folder, target_dir=save_root_dir/'train')
    mk_same_dir_structure(src_dir=my_clips_folder, target_dir=save_root_dir/'val')
    mk_same_dir_structure(src_dir=my_clips_folder, target_dir=save_root_dir/'test')

    all_mp4_paths = sorted(my_clips_folder.glob('**/*.mp4'))

    pose_inferencer = MMPoseInferencer('human')

    pbar = tqdm(range(len(all_mp4_paths)), desc='Yield .npy files', unit='video')
    for video_path in all_mp4_paths:
        # Set the save paths.
        ball_type_dir = video_path.parent
        vid, r, stroke_num = map(int, video_path.stem.split('_'))
        set_name = strokes_df.loc[
            (strokes_df['match'] == vid) &
            (strokes_df['rally'] == r) &
            (strokes_df['stroke'] == stroke_num),
            'set'
        ].item()
        save_branch = str(save_root_dir/set_name/ball_type_dir.name/video_path.stem)

        need_detect_players = not Path(save_branch+'_pos.npy').exists() or not Path(save_branch+'_joints.npy').exists()
        if need_detect_players:
            # Players detection
            players_positions, joints = \
                detect_players_2d(
                    inferencer=pose_inferencer,
                    vid=vid,
                    video_path=video_path,
                    res_df=resolution_df,
                    all_H=all_H,
                    normalized_by_v_height=joints_normalized_by_v_height,
                    center_align=joints_center_align
                )
            np.save(save_branch+'_pos.npy', players_positions)
            # (F, P, xy)
            np.save(save_branch+'_joints.npy', joints)
            # (F, P, J, xy)

        if not Path(save_branch+'_shuttle.npy').exists():
            # Get the shuttlecock position from the generated .csv files.
            shuttle_result_path = save_shuttle_dir/(video_path.stem+'_ball.csv')            
            shuttle_result = get_shuttle_result(
                path=shuttle_result_path,
                v_width=resolution_df.loc[vid, 'width'],
                v_height=resolution_df.loc[vid, 'height']
            )
            # shuttle_result: (F, 2)

            # Set the content of the frame failed in players detection to 0
            if not need_detect_players:
                players_positions = np.load(str(Path(save_branch+'_pos.npy')))
            failed_i = np.where(np.all(players_positions == 0, axis=(1, 2)))[0]
            if len(failed_i):
                shuttle_result[failed_i, :] = 0

            if len(players_positions) == len(shuttle_result):
                np.save(save_branch+'_shuttle.npy', shuttle_result)
                # (F, xy)
            else:
                print()
                print(f'{video_path.stem} time sequence {len(players_positions)} (pose) != {len(shuttle_result)} (shuttle)')
                shuttle_unmatch_ls.append(video_path.stem)
        # else:
        #     if not need_detect_players:
        #         players_positions = np.load(str(Path(save_branch+'_pos.npy')))
        #     shuttle_result = np.load(str(Path(save_branch+'_shuttle.npy')))

        #     if len(players_positions) != len(shuttle_result):
        #         print()
        #         print(f'{video_path.stem} time sequence {len(players_positions)} (pose) != {len(shuttle_result)} (shuttle)')
        #         shuttle_unmatch_ls.append(video_path.stem)

        pbar.update()
    pbar.close()
    return shuttle_unmatch_ls


def get_max_seq(npy_data_root: Path):
    npy_files = list(npy_data_root.glob('**/*_shuttle.npy'))
    ## .npy files aren't large.
    lengths = [len(np.load(f)) for f in npy_files]
    return max(lengths)
    ## If large, we can use the following code.
    # with ThreadPoolExecutor() as executor:
    #     lengths = executor.map(lambda f: len(np.load(f)), npy_files)
    # return max(lengths)


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


def handle_data_balance(
    data_arr_ls: list[np.ndarray],
    labels: np.ndarray,
    cnt_series: pd.Series
):
    max_n = cnt_series.max()
    label_max = labels.max()

    new_labels_ls = []
    new_data_ls_ls = []

    for i in range(label_max + 1):
        bool_idx: np.ndarray = labels == i
        n = bool_idx.sum()
        
        selected_data_ls = [d[bool_idx] for d in data_arr_ls]
        magni = max_n // n
        remain = max_n % n
        
        cur_new_labels = np.full(max_n, i, dtype=np.int64)

        cur_new_data_ls = [d.repeat(magni, axis=0) for d in selected_data_ls]
        if remain != 0:
            remain_data_ls = [d[:remain] for d in selected_data_ls]
            cur_new_data_ls = [np.concatenate(tup, axis=0) for tup in zip(cur_new_data_ls, remain_data_ls)]
        
        new_labels_ls.append(cur_new_labels)
        new_data_ls_ls.append(cur_new_data_ls)

    new_labels = np.concatenate(new_labels_ls, axis=0)
    new_data_arr_ls = [np.concatenate(tup, axis=0) for tup in zip(*new_data_ls_ls)]

    return_ls = [*new_data_arr_ls, new_labels]
    correct_len = max_n * (label_max + 1)
    for arr in return_ls:
        assert len(arr) == correct_len, 'Has an array\'s len != max_n.'
    return return_ls


def collate_npy(
    root_dir: Path,
    raw_data_root_dir: Path,
    set_name: str,
    seq_len: int,
    cnt_series: pd.Series,
    save_dir: Path
):
    '''Collate .npy data before to make training faster.
    Notice: This will pad the arrays to the same length.
    '''
    assert set_name in ['train', 'val', 'test'], 'Invalid set_name.'
    
    type_2_id = get_stroke_types(root_dir/'after_generating')

    # load .npy branch names
    data_branches = []
    labels = []
    target_dir = raw_data_root_dir/set_name
    for typ in target_dir.iterdir():
        shots = sorted([str(s).replace('_pos.npy', '') for s in typ.glob('*_pos.npy')])
        data_branches += shots
        labels.append(np.full(len(shots), type_2_id[typ.name], dtype=np.int64))
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

    if set_name == 'train':
        print('Data balancing ...')
        J_only, JnB_interp, JnB_bone, Jn2B, pos, shuttle, videos_len, labels \
            = handle_data_balance(
                data_arr_ls=[J_only, JnB_interp, JnB_bone, Jn2B, pos, shuttle, videos_len],
                labels=labels,
                cnt_series=cnt_series
            )
        print('Finish balancing.')

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
    badDB_root = Path('../BadmintonDB')
    res_path = badDB_root/'my_raw_video_resolution.csv'
    court_detection_dir = badDB_root/'court_detection'
    my_clips_folder = badDB_root/'set'

    badDB_data_root = Path("preparing_data/BadmintonDB_data")
    save_shuttle_dir = badDB_data_root/'shuttlecock_temp'
    save_root_dir_raw = badDB_data_root/'dataset_npy'
    save_root_dir_collated = badDB_data_root/'dataset_npy_balance_collated'

    resolution_df = pd.read_csv(res_path).set_index('id')
    all_corner_camera = get_corners_camera_all_videos(court_detection_dir)
    all_H = get_H_all_videos(all_corner_camera)

    # # Check if the homography matrices are correct
    # for vid, corner_camera in all_corner_camera.items():
    #     arr_court = to_court_coordinate(corner_camera.T, vid, all_H)
    #     all_close = np.allclose(arr_court.T, np.array([[0, 0], [0, 1], [1, 1], [1, 0]]), atol=1e-7)
    #     print(f'Video {vid}: {all_close}')

    strokes_df, cnt_df = split_dataset(badDB_root)
    print(cnt_df)

    while True:
        while True:
            ## Step 1
            prepare_trajectory(
                my_clips_folder=my_clips_folder,
                model_folder=Path("C:/MyResearch/TrackNetV3-main"),
                save_shuttle_dir=save_shuttle_dir
            )
            shuttles_need_try_again = check_empty_files_and_del(save_shuttle_dir)
            if len(shuttles_need_try_again) == 0:
                break
            print('Need try again empty csv files:', len(shuttles_need_try_again))

        ## Step 2
        shuttle_unmatch_ls = prepare_2d_dataset_npy_from_raw_video(
            my_clips_folder=my_clips_folder,
            save_shuttle_dir=save_shuttle_dir,
            save_root_dir=save_root_dir_raw,
            strokes_df=strokes_df,
            resolution_df=resolution_df,
            all_H=all_H,
            joints_normalized_by_v_height=False,
            joints_center_align=True
        )
        if len(shuttle_unmatch_ls) == 0:
            break
        del_csv_files(shuttle_unmatch_ls, root=save_shuttle_dir)
        print('Need try again time unmatch csv files:', len(shuttle_unmatch_ls))

    max_seq = get_max_seq(save_root_dir_raw)
    print('Max seq:', max_seq)

    ## Step 3
    for set_name in ['train', 'val', 'test']:
        collate_npy(
            root_dir=badDB_root,
            raw_data_root_dir=save_root_dir_raw,
            set_name=set_name,
            seq_len=72,  # from max_seq
            cnt_series=cnt_df['Train'][:-1],
            save_dir=save_root_dir_collated
        )
