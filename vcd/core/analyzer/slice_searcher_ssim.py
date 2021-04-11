import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from vcd.core.analyzer.slice_searcher import FramesSliceSearcher
from vcd.core.utils import make_template_jpg


class FramesSliceSearcherSSIM(FramesSliceSearcher):
    def __init__(self, video_name, frames):  # создание объекта поиска склеек на видео методом SSIM
        super().__init__(video_name, frames)
        self.__template = make_template_jpg(video_name)  # шаблон для поиска изображений

    def search_for_slices(self):
        """
            Метод для получения оценок
            дальше можно обрабатывать оценки любым удобным нам способом
            например, искать минимумы в массиве оценок и говорить, что в данном месте скорее всего была склейка
            или, все оценки, которые ниже некоторого порогового значения, тоже считать склейками
        """
        prev_image = cv2.imread(self.__template.format(self.__frames[0]))  # считываем первое изображение
        scores = []  # массив результатов анализа пар изображений
        for i in range(len(self.__frames) - 1):
            next_image = cv2.imread(self.__template.format(self.__frames[i + 1]))
            (score, diff) = ssim(prev_image, next_image, full=True, multichannel=True)
            prev_image = next_image  # текущее изображение делаем предыдущим
            scores.append(score)  # добававляем результат меры схожести в массив
        self.__scores = np.array(scores)

    def analyze_scores(self):
        mean = np.mean(self.__scores)
        var = np.sqrt(np.var(self.__scores))
        indexes = np.where(self.__scores < mean - 3 * var)
        return indexes, self.__scores[indexes]
