import os

import numpy as np

from vcd.analyzer.cut_detector_sift import CutDetectorSIFT
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
        slice_searcher = CutDetectorSIFT(path, local_th=75)  # limit = 5
        return get_scores(slice_searcher.search_for_slices(), prev)

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
