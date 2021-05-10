import json
import os

import numpy as np

from vcd.analyzer.score_collector_sift import ScoreCollectorSIFT
from vcd.analyzer.score.regression_model import load_model

VIDEO_PATH_KEY = 'video_path'
ALGORITHM_TYPE_KEY = "algorithm_type"
REGRESSION_MODEL_PATH_KEY = "lr_path"


class AlgorithmType:
    MSE = 0
    SSIM = 1
    SIFT = 2


def SIFT_CASE(config):
    offset = 30

    def build_scores(path, prev):
        slice_searcher = ScoreCollectorSIFT(path, local_th=75)  # limit = 5
        return get_scores(slice_searcher.collect(), prev)

    def get_scores(scores, prev):
        x = []
        for index, score in enumerate(scores):
            if index >= prev:
                x.append(scores[index - prev + 1: index + 1])
        return np.array(x)

    def resolve_regression_model_filename(conf):
        try:
            return conf[REGRESSION_MODEL_PATH_KEY]
        except KeyError:
            base_file = config[VIDEO_PATH_KEY]
            base_path = os.path.dirname(base_file)
            base_filename = os.path.splitext(base_file)[0]

            return os.path.join(base_path, base_filename + '.model')

    def resolve_time(indexes, fps):
        pass

    video_path = config[VIDEO_PATH_KEY]

    video_name = os.path.basename(video_path)
    folder_path = os.path.dirname(video_path)
    template_name = os.path.splitext(video_name)[0]

    X = np.array(build_scores(video_path, offset))
    # after generation X array, we have to try
    regression_model_filename = resolve_regression_model_filename(config)
    regression_model = load_model(regression_model_filename)
    # analyzer = RegressionModelScoreAnalyzer(X, )
    # todo: analyzer + calculate offset for each value on prediction
    return resolve_time(regression_model.predict(X) + offset + 1, 25)


def read_config_file(path):
    with open(path) as f:
        return json.load(f)


def create_config_file(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


if __name__ == '__main__':
    # video_path = input("Full path to video:")

    config_path = os.path.join(os.getcwd(), "vcd/resources/config/config.json")

    # config_data = {
    #     VIDEO_PATH_KEY: "C:/Users/ahumyck/PycharmProjects/diplom/vcd/resources/video/result.mp4",
    #     ALGORITHM_TYPE_KEY: AlgorithmType.MSE,
    #     REGRESSION_MODEL_PATH_KEY: ""
    # }
    #
    # create_config_file(config_data, config_path)

    config = read_config_file(config_path)
    video_path = config[VIDEO_PATH_KEY]
    video_name = os.path.basename(video_path)
    folder_path = os.path.dirname(video_path)
    template_name = os.path.splitext(video_name)[0]

    print("folder_path:", folder_path)
    print("template_name:", template_name)
    print("video_name:", video_name)
