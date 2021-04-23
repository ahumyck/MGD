import os

import numpy as np
import pandas as pd

from vcd.analyzer.cut_detector_sift import CutDetectorSIFT


def get_files(path_to_dir):
    return [os.path.join(path_to_dir, f) for f in os.listdir(path_to_dir) if
            os.path.isfile(os.path.join(path_to_dir, f))]


def build_scores(path_to_video):
    slice_searcher = CutDetectorSIFT(path_to_video, th=75)
    return get_scores(slice_searcher.search_for_slices())


def get_scores(scores, th=30):
    x = []

    for index, score in enumerate(scores):
        if index >= th:
            var = scores[index - th + 1: index + 1]
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
        target[frame] = 1
    return np.array(target)


if __name__ == '__main__':
    video_name = "/vcd/resources/video/result.mp4"
    target_filename = "/vcd/resources/data/target.txt"
    # todo: calculate offset as const
    offset = 31

    print(f'video with name {video_name}')
    X = np.array(build_scores(video_name))
    Y = np.array(build_target(target_filename, len(X), offset))

    dataframe = pd.DataFrame(
        {
            "frame average speed": X,
            "target": Y
        }
    )

    dataframe.to_excel("data.xlsx")
