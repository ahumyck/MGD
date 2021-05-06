import os

import numpy as np
import pandas as pd

from vcd.analyzer.cut_detector_sift import CutDetectorSIFT


def get_files(path_to_dir):
    return [os.path.join(path_to_dir, f) for f in os.listdir(path_to_dir) if
            os.path.isfile(os.path.join(path_to_dir, f))]


def build_scores(path_to_video, prev):
    slice_searcher = CutDetectorSIFT(path_to_video, local_th=75)  # limit = 5
    return get_scores(slice_searcher.search_for_slices(), prev)


def get_scores(scores, prev):
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
    training_data_filename = os.path.join(root, "vcd/resources/data/data.xlsx")
    video_name = os.path.join(root, "vcd/resources/video/result.mp4")
    target_filename = os.path.join(root, "vcd/resources/data/target.txt")

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
