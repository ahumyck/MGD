import abc

import numpy as np


class CutDetector:
    def __init__(self, video_name):  # создание объекта поиска склеек на видео методом SSIM
        self._video_name = video_name
        self._scores = np.array([])

    @abc.abstractmethod
    def search_for_slices(self):
        """
            Метод для получения оценок
            дальше можно обрабатывать оценки любым удобным нам способом
            например, искать минимумы в массиве оценок и говорить, что в данном месте скорее всего была склейка
            или, все оценки, которые ниже некоторого порогового значения, тоже считать склейками
        """
        return

    @abc.abstractmethod
    def analyze_scores(self):
        """
            Функция для поиска поиска склеек из полученных оценок
        :return: индексы предполагаемых склеек
        """
        return
