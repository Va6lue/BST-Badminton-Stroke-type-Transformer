import moviepy.editor as mpe
import pandas as pd
from pathlib import Path
import json


def get_txt_labels(folder: Path):
    df_list = []
    filenames = sorted(folder.glob('*'))

    for m, filename in enumerate(filenames, start=1):
        df = pd.read_csv(filename, sep='\s+', header=None, names=['frame_number', 'stroke_type'])
        df['event'] = (df['stroke_type'] != df['stroke_type'].shift()).cumsum()
        df = df.groupby(['event', 'stroke_type']).agg(
            start_frame=('frame_number', 'first'),
            end_frame=('frame_number', 'last')
        ).reset_index('stroke_type')
        df = df.loc[df['stroke_type'] != 'OTH'].reset_index()
        df.insert(0, 'match', m)
        df.insert(1, 'stroke', df.index + 1)

        df_list.append(df)
    
    df = pd.concat(df_list, ignore_index=True)
    return df


def get_json_split_info(folder: Path):
    '''Use the split information from their paper.'''
    df_list = []
    json_ls = sorted(folder.glob('*'))

    for m, json_info in enumerate(json_ls, start=1):
        with open(json_info, 'r', encoding='utf-8') as file:
            data: dict = json.load(file)
        
        split_info = data['classes']['SPLITS']
        for sp in split_info:
            df_one = pd.DataFrame({
                'match': [m],
                'dataset': sp['custom']['Type'],
                'start': sp['start'],
                'end': sp['end']
            })
            df_list.append(df_one)

    df = pd.concat(df_list, ignore_index=True)
    return df


def get_count_dataset_df(df: pd.DataFrame, col: str):
    train_df = df.loc[df['dataset'] == 'Train']
    val_df = df.loc[df['dataset'] == 'Val']
    test_df = df.loc[df['dataset'] == 'Test']
    
    train_cnt = train_df.groupby(col).size()
    val_cnt = val_df.groupby(col).size()
    test_cnt = test_df.groupby(col).size()

    cnt_df = pd.DataFrame({
        'Train': train_cnt,
        'Val': val_cnt,
        'Test': test_cnt,
        'Total': train_cnt + val_cnt + test_cnt
    })
    cnt_df.loc['Sum'] = cnt_df.sum()

    return cnt_df


def config_split(df: pd.DataFrame, split_info_df: pd.DataFrame):
    merged_df = pd.merge(df, split_info_df, on='match', how='left')

    merged_df['dataset'] = merged_df.apply(
        lambda row: row['dataset'] if (row['start'] <= row['start_frame'] and row['end_frame'] <= row['end']) else None,
        axis=1
    )

    df_config = merged_df.loc[merged_df['dataset'].notna()].reset_index(drop=True)
    return df_config


def gen_stroke_videos(
    raw_video_dir: Path,
    out_root_dir: Path,
    strokes_df: pd.DataFrame
):
    if not out_root_dir.is_dir():
        out_root_dir.mkdir()

    for s in strokes_df['dataset'].unique():
        s: str
        set_folder = out_root_dir/s
        if not set_folder.is_dir():
            set_folder.mkdir()
        for typ in strokes_df['stroke_type'].unique():
            typ: str
            stroke_folder = set_folder/typ
            if not stroke_folder.is_dir():
                stroke_folder.mkdir()

    v_paths = sorted(raw_video_dir.glob('*'))
    for m, v_path in enumerate(v_paths, start=1):
        video = mpe.VideoFileClip(str(v_path))
        
        df = strokes_df.loc[strokes_df['match'] == m]
        for row in df.itertuples(index=False):
            output_path = out_root_dir/row.dataset.lower()/row.stroke_type/f'{m}_{row.stroke}.mp4'
            if not output_path.exists():
                clip: mpe.VideoClip = video.subclip(row.start_frame / video.fps, row.end_frame / video.fps)
                clip.write_videofile(str(output_path))
        
        video.close()


def merge_types(df: pd.DataFrame, save=False):
    # Define merged types here.
    type_mapping = {
        'SFI': 'SF',  # Serve Far In -> Serve Far
        'SFL': 'SF',  # Serve Far Let -> Serve Far
        'SFF': 'SF',  # Serve Far Fault -> Serve Far
        'SNI': 'SN',  # Serve Near In -> Serve Near
        'SNL': 'SN',  # Serve Near Let -> Serve Near
        'SNF': 'SN'   # Serve Near Fault -> Serve Near
    }

    df = df.drop(columns=['start', 'end'])
    df.insert(4, 'merged_type', df['stroke_type'].replace(type_mapping))
    df = df.sort_values(by=['merged_type', 'match', 'stroke'])
    df['class_id'] = pd.factorize(df['merged_type'])[0]

    if save:
        df.to_csv('merged_strokes.csv', index=False)
    return df


if __name__ == "__main__":
    raw_video_dir = Path('videos')
    labels_dir = Path('labels')
    split_info_dir = Path('json')
    set_dir = Path('set')

    df = get_txt_labels(labels_dir)
    split_info_df = get_json_split_info(split_info_dir)

    df_config = config_split(df, split_info_df)
    cnt_df = get_count_dataset_df(df_config, col='stroke_type')
    print(cnt_df)

    gen_stroke_videos(
        raw_video_dir=raw_video_dir,
        out_root_dir=set_dir,
        strokes_df=df_config
    )

    df_merged = merge_types(df_config, save=True)
    cnt_df = get_count_dataset_df(df_merged, col='merged_type')
    print(cnt_df)
