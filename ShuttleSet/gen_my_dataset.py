from utils import *

import moviepy.editor as mpe
import pandas as pd
import numpy as np
from pathlib import Path


def get_video_df(set_info_dir: Path):
    df = pd.read_csv(set_info_dir/'match.csv')[['id', 'video', 'downcourt']]
    df['downcourt'] = df['downcourt'].astype(bool)
    return df.set_index('id')


def collect_shot_types_pos(
    set_info_dir: Path,
    v_info: pd.DataFrame,
    player: str,
    type_ls: list
):
    folder_path: Path = set_info_dir/v_info['video']
    first_A_is_top = v_info['downcourt']

    set_count = len(list(folder_path.glob('*.csv')))
    sets_collected_ls = []

    # For set1 and set2
    for i in range(1, 3):
        df = pd.read_csv(folder_path/f'set{i}.csv')
        df = df[['rally', 'ball_round', 'frame_num', 'roundscore_A', 'roundscore_B', 'player', 'type']]
        df = df[df['type'].isin(type_ls)]
        df.insert(0, 'set', np.full(len(df), i, int))

        if (first_A_is_top ^ (i == 2)):
            df['player'] = np.where(df['player'] == 'A', 'Top', 'Bottom')
        else:
            df['player'] = np.where(df['player'] == 'B', 'Top', 'Bottom')
        
        df = df[df['player'] == player]
        del df['player']

        sets_collected_ls.append(df)

    # 處理 11 分換場
    if set_count == 3:
        df = pd.read_csv(folder_path/'set3.csv')
        df = df[['rally', 'ball_round', 'frame_num', 'roundscore_A', 'roundscore_B', 'player', 'type']]
        df.insert(0, 'set', np.full(len(df), 3, int))

        # Compute the split point
        i = df['roundscore_A'].searchsorted(11, side='left')
        if df.loc[i, 'roundscore_A'] <= df.loc[i, 'roundscore_B']:
            i = df['roundscore_B'].searchsorted(11, side='left')
        i_split = df['rally'].searchsorted(df.loc[i, 'rally'], side='right')
        df_1 = df.iloc[:i_split]
        df_2 = df.iloc[i_split:]

        df_1 = df_1[df_1['type'].isin(type_ls)]
        df_2 = df_2[df_2['type'].isin(type_ls)]

        if first_A_is_top:
            df_1['player'] = np.where(df_1['player'] == 'A', 'Top', 'Bottom')
            df_2['player'] = np.where(df_2['player'] == 'B', 'Top', 'Bottom')
        else:
            df_1['player'] = np.where(df_1['player'] == 'B', 'Top', 'Bottom')
            df_2['player'] = np.where(df_2['player'] == 'A', 'Top', 'Bottom')

        df_1 = df_1[df_1['player'] == player]
        df_2 = df_2[df_2['player'] == player]
        del df_1['player']
        del df_2['player']

        sets_collected_ls += [df_1, df_2]

    return pd.concat(sets_collected_ls).reset_index(drop=True)


def set_between_2_hits_from_pos(folder_path: Path, shots_df: pd.DataFrame):
    new_df_ls = []

    groups = shots_df.groupby('set').groups
    for set_i, df_ids in groups.items():
        df = pd.read_csv(folder_path/f'set{set_i}.csv')
        df = df[['rally', 'ball_round', 'frame_num']]
        
        # get start points and end points
        df['start_f'] = df['frame_num'].shift(1)
        df['end_f'] = df['frame_num'].shift(-1)

        # Start point no former and End point no latter
        df['start_f'] = df['start_f'].where(df.duplicated('rally', keep='first'), -1)
        df['end_f'] = df['end_f'].where(df.duplicated('rally', keep='last'), -1)
        
        new_df = pd.merge(
            shots_df.iloc[df_ids].reset_index(drop=True),
            df,
            on=['rally', 'ball_round', 'frame_num']
        )
        new_df = new_df[[
            'set', 'rally', 'ball_round',
            'start_f', 'frame_num', 'end_f',
            'roundscore_A', 'roundscore_B', 'type'
        ]]
        new_df_ls.append(new_df)

    return pd.concat(new_df_ls).reset_index(drop=True)


def gen_shot_videos_from_1_src(
    raw_video_dir: Path,
    out_root_dir: Path,
    v_info: pd.Series,
    shots_df: pd.DataFrame,
    strategy: str,
    player: str,
    set_name: str,
    type_ls: list
):
    '''
    Parameters
    ----------
    `strategy`
    - middle_in_a_sec
    - between_2_hits
    - between_2_hits_with_max_limits
    '''
    if not out_root_dir.is_dir():
        out_root_dir.mkdir()
    
    out_folder = out_root_dir/set_name
    if not out_folder.is_dir():
        out_folder.mkdir()

    # Check type folders exist or not
    for typ in type_ls:
        sub_folder = out_folder/f'{player}_{typ}'
        if not sub_folder.is_dir():
            sub_folder.mkdir()

    video_path = str(next(raw_video_dir.glob(f"{v_info.name} *")))

    # MoviePy 版本
    video = mpe.VideoFileClip(video_path)
    t = video.fps // 2  # num frames in 0.5 sec

    match strategy:
        case 'middle_in_a_sec':
            for row in shots_df.itertuples(index=False):
                output_path = out_folder/f"{player}_{row.type}/{v_info.name}_{row.set}_{row.rally}_{int(row.ball_round)}.mp4"
                if not output_path.exists():
                    clip: mpe.VideoClip = video.subclip(
                        frameNum_2_time(int(row.frame_num) - t, fps=video.fps),
                        frameNum_2_time(int(row.frame_num) + t, fps=video.fps)
                    )
                    clip.write_videofile(str(output_path))

        case 'between_2_hits':
            eps = t // 2  # extension
            for row in shots_df.itertuples(index=False):
                start_f = int(row.start_f) if row.start_f != -1 else (int(row.frame_num) - t)
                end_f = int(row.end_f) + eps if row.end_f != -1 else (int(row.frame_num) + t)

                output_path = out_folder/f"{player}_{row.type}/{v_info.name}_{row.set}_{row.rally}_{int(row.ball_round)}.mp4"
                if not output_path.exists():
                    clip: mpe.VideoClip = video.subclip(
                        frameNum_2_time(start_f, fps=video.fps),
                        frameNum_2_time(end_f, fps=video.fps)
                    )
                    clip.write_videofile(str(output_path))

        case 'between_2_hits_with_max_limits':
            limit = video.fps * 3 // 2  # num frames in 1.5 sec
            eps = t // 2  # num frames in 0.25 sec
            for row in shots_df.itertuples(index=False):
                frame_num = int(row.frame_num)
                start_f = int(row.start_f) if row.start_f != -1 else (frame_num - t)
                end_f = int(row.end_f) + eps if row.end_f != -1 else (frame_num + t)

                # max limits
                if start_f < frame_num - limit:
                    start_f = frame_num - limit
                if end_f > frame_num + limit + eps:
                    end_f = frame_num + limit + eps

                output_path = out_folder/f"{player}_{row.type}/{v_info.name}_{row.set}_{row.rally}_{int(row.ball_round)}.mp4"
                if not output_path.exists():
                    clip: mpe.VideoClip = video.subclip(
                        frameNum_2_time(start_f, fps=video.fps),
                        frameNum_2_time(end_f, fps=video.fps)
                    )
                    clip.write_videofile(str(output_path))

        case _:
            raise NotImplementedError
    
    video.close()

    ## OpenCV 版本
    # cap = cv2.VideoCapture(video_path)
    # if not cap.isOpened():
    #     raise Exception('Error opening the video file.')
    
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # t = int(fps) // 2  # num frames in 0.5 sec
    # num_frames = t * 2 + 1

    # for i, row in shots_df.iterrows():
    #     output_path = out_folder/f"{player}_{row['type']}/{v_info.name}_{row['set']}_{row['rally']}_{int(row['ball_round'])}.mp4"
    #     if not output_path.exists():
    #         start_t = (int(row['frame_num']) - t) / fps
    #         write_video(str(output_path), cap, start_t=start_t, num_frames=num_frames)
    # cap.release()
    

if __name__ == "__main__":
    raw_video_dir = Path('raw_video')
    set_info_dir = Path('set')
    out_root_dir = Path('shuttle_set_between_2_hits_with_max_limits')

    v_df = get_video_df(set_info_dir)
    # print(v_df)
    vids_train = list(range(1, 9)) + [11] + list(range(13, 27)) + list(range(28, 35))
    vids_val = list(range(35, 39)) + [41]
    vids_test = [39, 40, 42, 43, 44]

    player = 'Top'  # Top / Bottom
    set_name = 'train'  # train / val / test

    type_ls = [
        '放小球', '擋小球', '殺球', '點扣', '挑球', '防守回挑',
        '長球', '平球', '小平球', '後場抽平球', '切球', '過渡切球', '推球',
        '撲球', '防守回抽', '勾球', '發短球', '發長球', '未知球種'
    ]

    match set_name:
        case 'train':
            vids_selected = vids_train
        case 'val':
            vids_selected = vids_val
        case 'test':
            vids_selected = vids_test

    for vid in vids_selected:
        v_info = v_df.loc[vid]
        # print(v_info)
        shots_df = collect_shot_types_pos(set_info_dir, v_info, player, type_ls)
        
        if len(shots_df) != 0:
            shots_df = set_between_2_hits_from_pos(
                folder_path=set_info_dir/v_info['video'],
                shots_df=shots_df
            )
            # with pd.option_context('display.max_rows', None):
            #     print(shots_df)
            gen_shot_videos_from_1_src(
                raw_video_dir=raw_video_dir,
                out_root_dir=out_root_dir,
                v_info=v_info,
                shots_df=shots_df,
                strategy='between_2_hits_with_max_limits',
                player=player,
                set_name=set_name,
                type_ls=type_ls
            )
