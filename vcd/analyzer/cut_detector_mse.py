import cv2
import numpy as np
from skimage.metrics import mean_squared_error as mse

from core.utils import make_template_jpg
from vcd.analyzer.cut_detector import CutDetector


class CutDetectorMSE(CutDetector):
    def __init__(self, video_name):  # создание объекта поиска склеек на видео методом MSE
        super().__init__(video_name)
        self.__template = make_template_jpg(video_name)

    def search_for_slices(self):
        """
            Метод для получения оценок
            дальше можно обрабатывать оценки любым удобным нам способом
            например, искать минимумы в массиве оценок и говорить, что в данном месте скорее всего была склейка
            или, все оценки, которые выше некоторого порогового значения, тоже считать склейками
        """
        capture = cv2.VideoCapture(self._video_name)  # получаем поток видео
        success, prev_image = capture.read()  # получаем очередное изображени е из видео

        scores = []

        while success:  # до тех пор, пока есть изображения в видео
            success, current_image = capture.read()  # берем следующее изображение
            if success:
                score = mse(prev_image, current_image)  # считаем меру схожести методом MSE
                prev_image = current_image  # текущее изображение делаем предыдущим
                scores.append(score)  # добававляем результат меры схожести в массив
            else:
                break

        capture.release()  # возвращаем ресурсы компьютеру
        self._scores = np.array(scores)

    def analyze_scores(self):
        mean = np.mean(self._scores)
        var = np.sqrt(np.var(self._scores))
        indexes = np.where(self._scores > mean + 3 * var)[0]
        return indexes, self._scores[indexes]
