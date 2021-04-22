import abc

import numpy as np
from sklearn.linear_model import LogisticRegression


class ScoreAnalyzer:
    def __init__(self, scores):
        self._scores = np.array(scores)

    @abc.abstractmethod
    def analyze(self):
        """
            Метод для анализа склеек на основе полученных результатов объекта CutDetector
            Возвращает пару (индексы, значения[индексы]) для потенциальных склеек
        :return:
        """
        return


class EmpiricalRuleScoreAnalyzer(ScoreAnalyzer):
    def __init__(self, scores):
        super().__init__(scores)

    def analyze(self):
        """
            среднее ± 3 сигма
        :return:
        """
        mean = np.mean(self._scores)
        sigma = np.sqrt(np.var(self._scores))
        more = np.where(self._scores > mean + 3 * sigma)[0]
        less = np.where(self._scores < mean - 3 * sigma)[0]
        indexes = np.concatenate([less, more])
        return indexes, self._scores[indexes]


class RegressionModelScoreAnalyzer(ScoreAnalyzer):
    def __init__(self, scores, model: LogisticRegression):
        super().__init__(scores)
        self.__model = model

    def analyze(self):
        # todo roc aug model
        return self.__model.predict(self._scores)
