import cv2
from skimage.metrics import structural_similarity as ssim

from core.utils import make_template_jpg
from vcd.analyzer.cut_detector import CutDetector


class CutDetectorSSIM(CutDetector):
    def __init__(self, video_name):  # создание объекта поиска склеек на видео методом SSIM
        super().__init__(video_name)
        self.__template = make_template_jpg(video_name)  # шаблон для поиска изображений

    def search_for_slices(self):
        """
            Метод для получения оценок
            дальше можно обрабатывать оценки любым удобным нам способом
            например, искать минимумы в массиве оценок и говорить, что в данном месте скорее всего была склейка
            или, все оценки, которые ниже некоторого порогового значения, тоже считать склейками
        """
        capture = cv2.VideoCapture(self._video_name)  # получаем поток видео
        success, prev_image = capture.read()  # получаем очередное изображени е из видео

        scores = []

        while success:  # до тех пор, пока есть изображения в видео
            success, current_image = capture.read()  # берем следующее изображение
            if success:
                (score, diff) = ssim(prev_image, current_image, full=True, multichannel=True)
                prev_image = current_image  # текущее изображение делаем предыдущим
                scores.append(score)  # добававляем результат меры схожести в массив
            else:
                break

        capture.release()  # возвращаем ресурсы компьютеру
        return scores
