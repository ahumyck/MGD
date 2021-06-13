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


def convert_scores(scores, slice_size):
    x = []

    for index, score in enumerate(scores):
        if index >= slice_size - 1:
            x.append(np.array(scores[index - slice_size + 1: index + 1]))

    return np.array(x)


# def convert_scores(scores, size):
#     expanded_scores = np.concatenate([[0] * (size - 1), scores])
#     x = []
#     for index, score in enumerate(expanded_scores):
#         if index >= size - 1:
#             vector = expanded_scores[index - size + 1: index + 1]
#             x.append(np.array(vector))
#     return np.array(x)


class RandomForestScoreAnalyzer(ScoreAnalyzer):
    def __init__(self, scores, model: LogisticRegression):
        super().__init__(scores)
        self.__model = model

    def analyze(self):
        indexes = np.where(self.__model.predict(self._scores) == 1)[0]
        return indexes, self._scores[indexes]  # indexes += (offset - 1)
