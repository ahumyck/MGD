import abc

import numpy as np


class ScoreAnalyzer:
    def __init__(self, scores):
        self._scores = np.array(scores)

    @abc.abstractmethod
    def analyze(self):
        """
            Метод для анализа склеек на основе полученных результатов объекта ScoreCollector
            Результатом работы метода является массив индексов, на которых были найдены подозрительные места
        :return:
        """
        return


def more_op(scores, mean, sigma):
    return np.where(scores > mean + 3 * sigma)


def less_op(scores, mean, sigma):
    return np.where(scores < mean - 3 * sigma)


class EmpiricalRuleScoreAnalyzer(ScoreAnalyzer):
    def __init__(self, scores, operations):
        super().__init__(scores)
        self.__ops = operations
        self.__mean = 0
        self.__sigma = 0

    def analyze(self):
        self.__mean = np.mean(self._scores)
        self.__sigma = np.sqrt(np.var(self._scores))

        result = []
        for op in self.__ops:
            result.append(op(self._scores, self.__mean, self.__sigma))
        indexes = np.concatenate(result)[0]
        return indexes, self._scores[indexes]

    def get_mean(self):
        return self.__mean

    def get_sigma(self):
        return self.__sigma
