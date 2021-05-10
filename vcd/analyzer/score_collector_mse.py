import cv2
from skimage.metrics import mean_squared_error as mse

from core.utils import make_template_jpg
from vcd.analyzer.score_collector import ScoreCollector


class ScoreCollectorMSE(ScoreCollector):
    def __init__(self, video_name):  # создание объекта поиска склеек на видео методом MSE
        super().__init__(video_name)
        self.__template = make_template_jpg(video_name)

    def collect(self):
        capture = cv2.VideoCapture(self._video_name)  # получаем поток видео
        success, prev_image = capture.read()  # получаем первое изображение из видео
        prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)  # делаем изображение чернобелым

        scores = []

        while success:  # до тех пор, пока есть изображения в видео
            success, current_image = capture.read()  # берем следующее изображение
            if success:
                current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)  # делаем изображение чернобелым
                score = mse(prev_image, current_image)  # считаем меру схожести методом MSE
                prev_image = current_image  # текущее изображение делаем предыдущим
                scores.append(score)  # добававляем результат меры схожести в массив
            else:
                break

        capture.release()  # возвращаем ресурсы компьютеру
        return scores
