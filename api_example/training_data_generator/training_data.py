import os
from os import listdir
from os.path import join, isfile

import numpy as np
import pandas as pd

from vcd.analyzer.score_collector_sift import ScoreCollectorSIFT


def get_videos(folder):
    return [f for f in listdir(folder) if isfile(join(folder, f))]


def get_files(path_to_dir):
    return [os.path.join(path_to_dir, f) for f in os.listdir(path_to_dir) if
            os.path.isfile(os.path.join(path_to_dir, f))]


def build_scores(path_to_video, prev):
    slice_searcher = ScoreCollectorSIFT(path_to_video, local_th=75)  # limit = 5
    return convert_scores(slice_searcher.collect(), prev)


def convert_scores(scores, prev):
    x = []

    for index, score in enumerate(scores):
        if index >= prev:
            var = scores[index - prev + 1: index + 1]
            x.append(str(var))

    return x


def parse_time_to_frame_with_delta(data, delta):
    time = data.replace("\n", "").split(":")
    return 25 * (int(time[0]) * 60 + int(time[1])) + int(time[2]) - delta


def build_target(filename, length, delta):
    frames = []

    with open(filename) as file:
        for line in file:
            frame = parse_time_to_frame_with_delta(line, delta)
            frames.append(frame)

    target = [0] * length
    for frame in frames:
        if frame >= length:
            break
        target[frame] = 1
    return np.array(target)


if __name__ == '__main__':
    root = os.getcwd()
    training_data_filename = os.path.join(root, "vcd/resources/training/data/data.xlsx")
    video_name = os.path.join(root, "vcd/resources/training/video/result.mp4")
    target_filename = os.path.join(root, "vcd/resources/training/video/target.txt")

    prev = 30
    offset = prev + 1

    print(f'video with name {video_name}')
    X = np.array(build_scores(video_name, prev))
    Y = np.array(build_target(target_filename, len(X), offset))

    dataframe = pd.DataFrame(
        {
            "frame average speed": X,
            "target": Y
        }
    )

    dataframe.to_excel(training_data_filename)

    path_to_videos = 'C:/Users/ahumyck/PycharmProjects/diplom/vcd/resources/MV'
    video_names = get_videos(path_to_videos)
    offset = 30

    for video_name in video_names:
        if video_name in ["2.mp4"]:
            continue
        full_video_path = os.path.join(path_to_videos, video_name)
        print(f'current video = {full_video_path}')
        X = np.array(build_scores(full_video_path, offset))

        video_template = os.path.splitext(video_name)[0]
        xlsx_template = os.path.join(path_to_videos, 'data' + video_template + '.xlsx')

        dataframe = pd.DataFrame(
            {
                "frame average speed": X,
            }
        )

        dataframe.to_excel(xlsx_template)
