import moviepy.editor as mpe
import pandas as pd
import numpy as np
from pathlib import Path
import json


def get_videos_json_info(json_info_dir: Path):
    df_list = []
    match_ls = sorted(json_info_dir.glob('*'))

    for m, json_info in enumerate(match_ls, start=1):
        if len(match_ls) == 9:
            if m == 4:  # Strokes in video 4 are all wrong.
                continue
        else:
            if m > 3:
                m += 1

        with open(json_info, 'r', encoding='utf-8') as file:
            data: dict = json.load(file)

        for r, point in enumerate(data.values(), start=1):
            point: dict = point['PointInfo']
            T1P1_is_top = point['T1P1']['Location'] == 'top'
            rally: list = point['Rally']

            flick_serve_first = False
            for stroke in rally:
                stroke: dict

                if 'StrokeType' in stroke.keys():
                    # if (stroke['StrokeType'] == 'Flick-Serve'):
                    if (stroke['StrokeType'] == 'Flick-Serve' and \
                        (m != 9 or r != 18 or stroke['StrokeNum'] != 2)):
                        # Flick-Serve always starts from the second stroke, which causes label errors.
                        # I found that decreasing the stroke number by one and swaping top/bottom can easily solve this problem.
                        flick_serve_first = True
                        T1P1_is_top = not T1P1_is_top

                    if (stroke['StrokeType'] != 'FAULT' and \
                        stroke['Camera'] == "normal" and \
                        stroke['StrokeBegin'] != None):
                        # Pick the strokes that are usable.

                        if stroke['StrokeType'][-3:] == '-Bh':
                            s_type = stroke['StrokeType'][:-3]
                        else:
                            s_type = stroke['StrokeType']

                        if (stroke['Player'] == 'T2P1') ^ T1P1_is_top:
                            s_type = 'Top-' + s_type
                        else:
                            s_type = 'Bottom-' + s_type

                        df_one = pd.DataFrame({
                            'match': [m],
                            'rally': [r],
                            'stroke': [stroke['StrokeNum'] - int(flick_serve_first)],
                            'stroke_type': [s_type],
                            'begin': [stroke['StrokeBegin']],
                            'end': [stroke['StrokeEnd']]
                        })
                        df_list.append(df_one)

    df = pd.concat(df_list, ignore_index=True)
    return df


def extra_correct_rally(ori_df: pd.DataFrame, need_corrected_df: pd.DataFrame):
    '''Decrease the stroke numbers by one and swap top/Bottom.'''
    df = pd.merge(
        ori_df,
        need_corrected_df,
        on=['match', 'rally'],
        how='left',
        indicator=True
    )
    mask = df['_merge'] == 'both'

    df.loc[mask, 'stroke'] -= 1
    df.loc[mask, 'stroke_type'] = np.where(
        df.loc[mask, 'stroke_type'].str.contains('Bottom'),
        df.loc[mask, 'stroke_type'].str.replace('Bottom', 'Top'),
        df.loc[mask, 'stroke_type'].str.replace('Top', 'Bottom')
    )
    return df.drop(columns=['_merge'])


def extra_exclude_strokes(ori_df: pd.DataFrame, need_deleted_df: pd.DataFrame):
    df = pd.merge(
        ori_df,
        need_deleted_df,
        on=['match', 'rally', 'stroke', 'stroke_type'],
        how='left',
        indicator=True
    )
    df = df.loc[df['_merge'] == 'left_only'].reset_index(drop=True).drop(columns=['reason', '_merge'])
    assert len(df) == len(ori_df) - len(need_deleted_df), "Some need deleted strokes are not in the original dataframe."
    return df


def gen_stroke_videos(
    raw_video_dir: Path,
    out_root_dir: Path,
    strokes_df: pd.DataFrame
):
    if not out_root_dir.is_dir():
        out_root_dir.mkdir()

    for typ in strokes_df['stroke_type'].unique():
        typ: str
        sub_folder = out_root_dir/typ
        if not sub_folder.is_dir():
            sub_folder.mkdir()

    v_paths = sorted(raw_video_dir.glob('*'))
    for m, v_path in enumerate(v_paths, start=1):
        video = mpe.VideoFileClip(str(v_path))
        
        df = strokes_df.loc[strokes_df['match'] == m]
        for row in df.itertuples(index=False):
            output_path = out_root_dir/row.stroke_type/f'{m}_{row.rally}_{row.stroke}.mp4'
            if not output_path.exists():
                clip: mpe.VideoClip = video.subclip(row.begin, row.end)
                clip.write_videofile(str(output_path))
        
        video.close()


def del_file(p: Path):
    p.unlink(missing_ok=True)


def del_generated_but_should_be_excluded_stroke_videos(set_dir: Path, need_deleted_df: pd.DataFrame):
    '''Use to delete the videos that need to be deleted but have already been generated.'''
    file_paths: pd.Series = need_deleted_df.apply(
        lambda row: set_dir/f'{row.stroke_type}/{row.match}_{row.rally}_{row.stroke}.mp4',
        axis=1
    )
    file_paths.apply(del_file)


def merge_types_and_save_df(df: pd.DataFrame, root_dir=Path('after_generating')):
    merged_df = pd.read_csv(root_dir/'merged_stroke_types.csv').drop(columns=['reason'])
    type_mapping = dict(zip(merged_df['stroke_type'], merged_df['become_stroke_type']))

    df = df.drop(columns=['begin', 'end'])
    df['merged_type'] = df['stroke_type'].replace(type_mapping)
    df = df.sort_values(by=['merged_type', 'match', 'rally', 'stroke'])
    df['class_id'] = pd.factorize(df['merged_type'])[0]

    # Count each merged class numbers
    type_count = df.groupby(['merged_type']).size()
    print(type_count)
    print('Total strokes:', type_count.sum())

    df.to_csv(root_dir/'merged_strokes.csv', index=False)


if __name__ == "__main__":
    raw_video_dir = Path('raw_video')
    json_info_dir = Path('json')
    set_dir = Path('set')
    
    df = get_videos_json_info(json_info_dir)

    need_corrected_df = pd.read_csv('before_generating/extra_rally_correction.csv')
    df = extra_correct_rally(df, need_corrected_df)

    need_deleted_df = pd.read_csv('before_generating/extra_excluded_strokes.csv')
    df = extra_exclude_strokes(df, need_deleted_df)

    # Count each class numbers
    type_count = df.groupby(['stroke_type']).size()
    print(type_count)
    print('Total strokes:', type_count.sum())

    gen_stroke_videos(
        raw_video_dir=raw_video_dir,
        out_root_dir=set_dir,
        strokes_df=df
    )

    # need_deleted_df = pd.read_csv('before_generating/extra_excluded_strokes.csv')
    # del_generated_but_should_be_excluded_stroke_videos(set_dir, need_deleted_df)

    merge_types_and_save_df(df)
