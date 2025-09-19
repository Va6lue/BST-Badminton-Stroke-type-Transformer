import cv2
from typing import Union


def time_2_frameNum(time_str, fps: Union[int, float]):
    """
    Converts time data (string or numeric components) to frame number based on FPS.

    Parameters
    ----------
    `time_str`
        Can be a string in 'H:M:S.s' or 'M:S.s' format, or numeric values for (hours, minutes, seconds).
    `fps`
        The frames per second of the video.

    Return
    ------
    The frame number (started from 0).
    """

    if isinstance(time_str, str):
        if time_str.count(':') == 1:
            time_str = '0:' + time_str
        try:
            hour, minute, second = map(float, time_str.split(':'))
        except ValueError:
            raise Exception('Invalid time string format')
    else:
        hour, minute, second = time_str

    total_seconds = hour * 3600 + minute * 60 + second
    return int(fps * total_seconds)


def frameNum_2_time(frame_number: int, fps: Union[int, float]) -> str:
    """
    Converts a frame number to time string (HH:MM:SS.ssssss) based on FPS.

    Parameters
    ----------
    `frame_number`
        The frame number (started from 0).
    `fps`
        The frames per second of the video.

    Returns
    -------
    The time string corresponding to the frame number.
    """
    total_seconds = frame_number / fps

    hours = int(total_seconds // 3600)
    minutes = int(total_seconds % 3600 // 60)
    seconds = total_seconds % 60 + 0.5 / fps

    return f"{hours:02d}:{minutes:02d}:{seconds:09.6f}"


def is_time_timeStr_convert_correct(test_frame_len: int, fps: Union[int, float]):
    '''
    Check whether converting frame numbers to time strings and then converting them back are correct or not.
    '''
    for i in range(test_frame_len):
        t_str = frameNum_2_time(i, fps)
        if i != time_2_frameNum(t_str, fps):
            print(f'Errors start at frame {i}.')
            return False
    return True


def write_video(output_path: str, cap: cv2.VideoCapture, start_t, num_frames: int):
    '''Write .mp4 video.'''
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_t * 1000)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:  # Check if frame is successfully read
            raise Exception('cap read failed.')
        video_writer.write(frame)
    video_writer.release()


if __name__ == '__main__':
    pass
