import cv2
from skimage.metrics import structural_similarity as ssim

from vcd.analyzer.score_collector import ScoreCollector


class ScoreCollectorSSIM(ScoreCollector):
    def __init__(self, video_name):
        super().__init__(video_name)

    def collect(self):
        capture = cv2.VideoCapture(self._video_name)  # получаем поток видео
        success, prev_image = capture.read()  # получаем очередное изображение из видео

        scores = []

        while success:  # до тех пор, пока есть изображения в видео
            success, current_image = capture.read()  # берем следующее изображение
            if success:
                (score, diff) = ssim(prev_image, current_image, full=True, multichannel=True)
                prev_image = current_image  # текущее изображение делаем предыдущим
                scores.append(score)  # добававляем результат меры схожести в массив

        capture.release()  # возвращаем ресурсы
        return scores
