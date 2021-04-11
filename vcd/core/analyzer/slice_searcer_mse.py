import cv2
import numpy as np
from skimage.metrics import mean_squared_error as mse

from vcd.core.analyzer.slice_searcher import FramesSliceSearcher
from vcd.core.utils import make_template_jpg


class FramesSliceSearcherMSE(FramesSliceSearcher):
    def __init__(self, video_name, frames):  # создание объекта поиска склеек на видео методом MSE
        super().__init__(video_name, frames)
        self.__template = make_template_jpg(video_name)

    def search_for_slices(self):
        """
            Метод для получения оценок
            дальше можно обрабатывать оценки любым удобным нам способом
            например, искать минимумы в массиве оценок и говорить, что в данном месте скорее всего была склейка
            или, все оценки, которые выше некоторого порогового значения, тоже считать склейками
        """
        prev_image = cv2.imread(self.__template.format(self.__frames[0]))  # считываем первое изображение
        scores = []  # массив результатов анализа пар изображений
        for i in range(len(self.__frames) - 1):
            next_image = cv2.imread(self.__template.format(self.__frames[i + 1]))  # считываем следующее изображение
            score = mse(prev_image, next_image)  # считаем меру схожести методом MSE
            prev_image = next_image  # текущее изображение делаем предыдущим
            scores.append(score)  # добававляем результат меры схожести в массив
        self.__scores = np.array(scores)

    def analyze_scores(self):
        mean = np.mean(self.__scores)
        var = np.sqrt(np.var(self.__scores))
        indexes = np.where(self.__scores > mean + 3 * var)
        return indexes, self.__scores[indexes]
