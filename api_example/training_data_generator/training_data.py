import os

import numpy as np
import pandas as pd

# from core.utils import get_frames_and_length_of_video
from vcd.analyzer.cut_detector_mse import CutDetectorMSE
from vcd.analyzer.cut_detector_sift import CutDetectorSIFT, INVALID_VALUE
from vcd.analyzer.score.score_analyzer import EmpiricalRuleScoreAnalyzer


def get_files(path_to_dir):
    return [os.path.join(path_to_dir, f) for f in os.listdir(path_to_dir) if
            os.path.isfile(os.path.join(path_to_dir, f))]


def build_target(path_to_video):
    slice_searcher = CutDetectorMSE(path_to_video)
    scores = slice_searcher.search_for_slices()  # получение результатов
    score_analyzer = EmpiricalRuleScoreAnalyzer(scores)
    indexes, _ = score_analyzer.analyze()
    return get_target(indexes, len(scores))


def build_scores(path_to_video):
    slice_searcher = CutDetectorSIFT(path_to_video)
    return get_scores(slice_searcher.search_for_slices())


def get_scores(scores, th=30):
    x = []

    for index, score in enumerate(scores):
        if index >= th:
            var = np.array(scores[index - th + 1: index + 1])
            var = var[var != INVALID_VALUE]
            x.append(str(list(var)))

    return x


def get_target(indexes, length, th=30):
    y = []

    for index in range(length):
        if index >= th:
            if index in indexes:
                y.append(1)
            else:
                y.append(0)

    return y


if __name__ == '__main__':
    video_name = "C:\\Users\\ahumyck\\PycharmProjects\\diplom\\vcd\\resources\\video\\result.mp4"
    X = []
    Y = []

    print(f'video with name {video_name}')
    X.extend(build_scores(video_name))
    print("scores done for video")
    Y.extend(build_target(video_name))
    print("target done for video")

    Y = np.array(Y)
    X = np.array(X)

    dataframe = pd.DataFrame(
        {
            "v(t) series": np.array(X),
            "y series": np.array(Y)
        }
    )

    dataframe.to_excel("output_first.xlsx")
