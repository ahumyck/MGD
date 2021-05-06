import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression

from vcd.analyzer.score.score_analyzer import ScoreAnalyzer


def load_model(filename) -> LogisticRegression:
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_mode(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


# todo: regression model
class RegressionModelScoreAnalyzer(ScoreAnalyzer):
    def __init__(self, scores, offset, model: LogisticRegression):
        super().__init__(scores)
        self.__model = model
        self.__ofset = offset

    def analyze(self):
        return np.where(self.__model.predict(self._scores) == 1)
