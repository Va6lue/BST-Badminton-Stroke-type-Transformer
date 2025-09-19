from mmpose.apis import MMPoseInferencer

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import subprocess
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor

from tenniSet_dataset import get_stroke_types, get_bone_pairs, make_seq_len_same, create_bones, interpolate_joints


def detect_ball_by_TrackNetV3_with_attension(
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


def detect_court_by_TennisCourtDetector(
    cur_i: int,
    total_tasks: int,
    video_path: Path,
    save_dir: Path,
    model_folder=Path("../TennisCourtDetector"),
):
    '''
    TennisCourtDetector
    https://github.com/yastrebksv/TennisCourtDetector
    '''
    process_args = [
        'python', str(model_folder/'infer_in_video.py').replace('\\', '/'),
        "--model_path", str(model_folder/"weight"/"model_tennis_court_det.pt").replace('\\', '/'),
        "--input_path", str(video_path).replace('\\', '/'),
        "--output_path", str(save_dir/video_path.stem).replace('\\', '/'),
        "--use_refine_kps",
        "--use_homography",
        # "--output_video"  # added myself
    ]
    r = subprocess.run(process_args)
    assert r.returncode == 0, 'Subprocess failed!'

    type_path = video_path.parent
    set_name = type_path.parent.name
    print(f'Court detection ({cur_i}/{total_tasks}): {set_name}/{type_path.name}/{video_path.name} done!')


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
                    detect_ball_by_TrackNetV3_with_attension,
                    i, len(all_mp4_paths),
                    video_path=video_path,
                    save_dir=save_shuttle_dir,
                    model_folder=model_folder
                )


def prepare_court_coordinates(
    my_clips_folder: Path,
    model_folder: Path,
    save_court_dir: Path,
):
    '''Court detection

    Notice: max_workers shouldn't be too high because this process is using GPU as well.
    '''
    if not save_court_dir.is_dir():
        save_court_dir.mkdir()

    all_mp4_paths = sorted(my_clips_folder.glob('**/*.mp4'))

    with ProcessPoolExecutor(max_workers=4) as executor:
        for i, video_path in enumerate(all_mp4_paths, start=1):
            court_result_path = save_court_dir/(video_path.stem+'.csv')
            if not court_result_path.exists():
                executor.submit(
                    detect_court_by_TennisCourtDetector,
                    i, len(all_mp4_paths),
                    video_path=video_path,
                    save_dir=save_court_dir,
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


def convert_homogeneous(arr: np.ndarray):
    '''
    The shape of `arr` is (..., 2, N). => The output will be (..., 3, N).
    '''
    shape = list(arr.shape)
    shape[-2] = 1
    return np.concatenate((arr, np.full(shape, 1.0)), axis=-2)


def project(H: np.ndarray, P_prime: np.ndarray):
    '''
    Transform coordinates from the camera system to the court system.
    
    H: (..., 3, 3)
    P_prime: (..., 3, N)
    Output: (..., 2, N)
    '''
    P = H @ P_prime
    P = P[..., :2, :] / P[..., -1:, :]  # /= w
    return P


def to_court_coordinate(
    arr_camera: np.ndarray,
    H: np.ndarray
):
    '''
    Convert the camera coordinate system to the court coordinate system.

    The shape of `arr_camera` is (..., 2, N).
    '''
    arr_camera = convert_homogeneous(arr_camera)
    arr_court = project(H, arr_camera)
    return arr_court


def normalize_joints(
    arr: np.ndarray,
    bbox: np.ndarray,
    v_height=None,
    center_align=True,
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


def find_out_court_player(
    pos_court_normalized: np.ndarray,
    find_far=True
):
    x_tolerance = 0.15
    pos_x = pos_court_normalized[:, 0]
    pos_y = pos_court_normalized[:, 1]
    pos_y = np.where((pos_x > -x_tolerance) & (pos_x < 1 + x_tolerance), pos_y, 0.5)

    if find_far:
        d = np.abs(pos_y)
    else:
        d = np.abs(pos_y - 1)
    
    return np.argmin(d)


def pick_players_from_human(
    keypoints: np.ndarray,
    H: np.ndarray
):
    '''
    The shape of `keypoints` is (m, J, 2).

    Output:
        is_player: (m)
        pos_court_normalized: (m, 2)
    '''
    n_people = keypoints.shape[0]
    
    feet_camera = keypoints[:, -2:, :]
    # feet_camera: (m, J, 2), J=2
    feet_camera = feet_camera.reshape(-1, 2).T
    # feet_camera: (2, m*J)

    feet_court = to_court_coordinate(feet_camera, H)
    feet_court = feet_court.reshape(2, n_people, -1)
    # feet_court: (2, m, J)

    pos_court_normalized = feet_court.mean(axis=-1).T  # middle point between feet
    pos_court_normalized: np.ndarray
    # pos_court_normalized: (m, 2)
    
    eps = 0.01  # soft border
    dim_in_court = (pos_court_normalized > -eps) & (pos_court_normalized < (1 + eps))
    in_court = dim_in_court[:, 0] & dim_in_court[:, 1]
    # in_court: (m)

    match in_court.sum():
        case 0:
            pid = find_out_court_player(pos_court_normalized, find_far=True)
            in_court[pid] = True
            pid = find_out_court_player(pos_court_normalized, find_far=False)
            in_court[pid] = True
        case 1:
            in_court_pid = np.nonzero(in_court)[0][0]
            need_find_far = pos_court_normalized[in_court_pid][1] > 0.5
            pid = find_out_court_player(pos_court_normalized, find_far=need_find_far)
            in_court[pid] = True
        case 2:
            pass

    return in_court, pos_court_normalized


def detect_players_2d(
    inferencer: MMPoseInferencer,
    video_path: Path,
    court_failed_i: np.ndarray,
    Hs: list[np.ndarray],
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

        # The court detection should be successful in this frame.
        # There should be at least 2 people in the video frame.
        failed = court_failed_i[frame_num] or len(keypoints) < 2
        if not failed:
            is_player, pos_normalized = pick_players_from_human(keypoints, Hs[frame_num])
            # is_player: (m), pos_normalized: (m, xy), xy=2
            is_player_id = np.nonzero(is_player)[0]
            
            # There should be 2 players only in a normal case.
            # If < 2 players, court detection probably failed.
            failed = len(is_player_id) != 2
            if not failed:
                bboxes = np.array([person['bbox'][0]
                                    for person in result['predictions'][0]])  # batch_size=1 (default)
                # bboxes: (m, 4)

                # Make sure Top player before Bottom player (comparing y-dim)
                if pos_normalized[is_player_id[0], 1] > pos_normalized[is_player_id[1], 1]:
                    is_player_id = np.flip(is_player_id)
                
                players_positions.append(pos_normalized[is_player_id])
                players_joints.append(normalize_joints(
                    arr=keypoints[is_player_id],
                    bbox=bboxes[is_player_id],
                    v_height=720 if normalized_by_v_height else None,
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


def read_court_csv(p: Path, corners_court: np.ndarray, return_Hs=True):
    df = pd.read_csv(p)
    corners_camera = df.drop(columns='id').to_numpy(na_value=0).reshape(-1, 4, 2)
    failed_i = np.any(corners_camera == 0, axis=(1, 2))
    Hs = None
    if return_Hs:
        Hs = [cv2.findHomography(arr, corners_court)[0] for arr in corners_camera]
    return failed_i, Hs


def prepare_2d_dataset_npy_from_raw_video(
    my_clips_folder: Path,
    save_shuttle_dir: Path,
    save_court_dir: Path,
    save_root_dir: Path,
    joints_normalized_by_v_height=False,
    joints_center_align=True
):
    shuttle_unmatch_ls = []

    mk_same_dir_structure(src_dir=my_clips_folder, target_dir=save_root_dir)

    corners_court = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

    all_mp4_paths = sorted(my_clips_folder.glob('**/*.mp4'))

    pose_inferencer = MMPoseInferencer('human')

    pbar = tqdm(range(len(all_mp4_paths)), desc='Yield .npy files', unit='video')
    for video_path in all_mp4_paths:
        # Set the save paths.
        ball_type_dir = video_path.parent
        set_split_dir = ball_type_dir.parent
        save_branch = str(save_root_dir/set_split_dir.name/ball_type_dir.name/video_path.stem)

        need_detect_players = not Path(save_branch+'_pos.npy').exists() or not Path(save_branch+'_joints.npy').exists()
        if need_detect_players:
            # Get the court corners and compute homography matrices each frame.
            failed_i, Hs = read_court_csv(
                p=save_court_dir/(video_path.stem + '.csv'),
                corners_court=corners_court,
                return_Hs=True
            )
            # Players detection
            players_positions, joints = \
                detect_players_2d(
                    inferencer=pose_inferencer,
                    video_path=video_path,
                    court_failed_i=failed_i,
                    Hs=Hs,
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
                v_width=1280,
                v_height=720
            )
            # shuttle_result: (F, 2)

            if not need_detect_players:
                failed_i, _ = read_court_csv(
                    p=save_court_dir/(video_path.stem + '.csv'),
                    corners_court=corners_court,
                    return_Hs=False
                )
            shuttle_result[failed_i, :] = 0

            if len(players_positions) == len(shuttle_result):
                np.save(save_branch+'_shuttle.npy', shuttle_result)
                # (F, xy)
            else:
                print()
                print(f'{video_path.stem} time sequence {len(players_positions)} (pose) != {len(shuttle_result)} (shuttle)')
                shuttle_unmatch_ls.append(video_path.stem)

        pbar.update()
    pbar.close()
    return shuttle_unmatch_ls


def get_max_seq(npy_data_root: Path):
    npy_files = list(npy_data_root.glob('**/*_shuttle.npy'))
    max_len_not_serve = 0
    max_file_not_serve = None
    
    with ThreadPoolExecutor() as executor:
        lengths = list(executor.map(lambda f: len(np.load(f)), npy_files))

    for file, length in zip(npy_files, lengths):
        if 'S' not in file.parent.stem and max_len_not_serve < length:
            max_file_not_serve = file
            max_len_not_serve = length
    
    return max_file_not_serve, max_len_not_serve, max(lengths)


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


def collate_npy(
    root_dir: Path,
    raw_data_root_dir: Path,
    set_name: str,
    seq_len: int,
    save_dir: Path
):
    '''Collate .npy data before to make training faster.
    Notice: This will pad the arrays to the same length.
    '''
    assert set_name in ['train', 'val', 'test'], 'Invalid set_name.'
    
    type_2_id = get_stroke_types(root_dir)

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


if __name__ == "__main__":
    tenniSet_root = Path('../TenniSet')
    my_clips_folder = tenniSet_root/'set'

    tenniSet_data_root = Path("preparing_data/TenniSet_data")
    save_court_dir = tenniSet_data_root/'court_temp'
    save_shuttle_dir = tenniSet_data_root/'ball_temp'
    save_root_dir_raw = tenniSet_data_root/'dataset_npy'
    save_root_dir_collated = tenniSet_data_root/'dataset_npy_collated'

    n_data = len(pd.read_csv(tenniSet_root/"merged_strokes.csv"))

    while True:
        ## Step 1
        prepare_court_coordinates(
            my_clips_folder=my_clips_folder,
            model_folder=Path('../TennisCourtDetector'),
            save_court_dir=save_court_dir
        )
        n_court_data = len(list(save_court_dir.glob('*')))
        if n_data == n_court_data:
            break
        print('Need try again court csv files:', n_data - n_court_data)

    while True:
        while True:
            ## Step 2
            prepare_trajectory(
                my_clips_folder=my_clips_folder,
                model_folder=Path("C:/MyResearch/TrackNetV3-main"),
                save_shuttle_dir=save_shuttle_dir
            )
            shuttles_need_try_again = check_empty_files_and_del(save_shuttle_dir)
            if len(shuttles_need_try_again) == 0:
                break
            print('Need try again empty csv files:', len(shuttles_need_try_again))

        ## Step 3
        shuttle_unmatch_ls = prepare_2d_dataset_npy_from_raw_video(
            my_clips_folder=my_clips_folder,
            save_shuttle_dir=save_shuttle_dir,
            save_court_dir=save_court_dir,
            save_root_dir=save_root_dir_raw,
            joints_normalized_by_v_height=False,
            joints_center_align=True
        )
        if len(shuttle_unmatch_ls) == 0:
            break
        del_csv_files(shuttle_unmatch_ls, root=save_shuttle_dir)
        print('Need try again time unmatch csv files:', len(shuttle_unmatch_ls))

    max_filename_not_serve, max_seq_not_serve, max_seq = get_max_seq(save_root_dir_raw)
    print('(Not serve) Max seq filename:', max_filename_not_serve)
    print('(Not serve) Max seq:', max_seq_not_serve)
    print('Max seq:', max_seq)

    ## Step 4
    for set_name in ['train', 'val', 'test']:
        collate_npy(
            root_dir=tenniSet_root,
            raw_data_root_dir=save_root_dir_raw,
            set_name=set_name,
            seq_len=100,
            save_dir=save_root_dir_collated
        )
