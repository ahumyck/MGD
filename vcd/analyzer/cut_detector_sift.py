import cv2
import numpy as np

from core.matcher import match_by_bruteforce_fast, sortMatchersByNorm, average_for_matchers
from vcd.analyzer.cut_detector import CutDetector


class CutDetectorSIFT(CutDetector):
    def __init__(self, video_name):
        super().__init__(video_name)
        self.sift = cv2.SIFT_create()
        self._possible_cut = []

    def search_for_slices(self):
        capture = cv2.VideoCapture(self._video_name)  # получаем поток видео
        success, prev_image = capture.read()  # получаем очередное изображени е из видео
        kp_prev, des_prev = self.sift.detectAndCompute(prev_image, None)

        scores = []
        n = 0

        while success:  # до тех пор, пока есть изображения в видео
            success, current_image = capture.read()  # берем следующее изображение
            if success:
                print(n)
                kp_current, des_current = self.sift.detectAndCompute(current_image, None)

                # ищем совпадающие ключевые точки
                matchers = match_by_bruteforce_fast(kp_prev, des_prev, kp_current, des_current)
                if len(matchers) == 0:
                    self._possible_cut.append(n)
                else:
                    # сортируем
                    matchers = sortMatchersByNorm(matchers)
                    # берем 5 "лучших" точек
                    matchers = matchers[0:5]
                    # считаем среднее движение кадра
                    scores.append(np.mean(average_for_matchers(matchers, kp_prev, kp_current)))
                kp_prev = kp_current
                des_prev = des_current
                n += 1

            else:
                break

        capture.release()  # возвращаем ресурсы компьютеру
        self._scores = np.array(scores)

    def analyze_scores(self):
        mean = np.mean(self._scores)
        var = np.sqrt(np.var(self._scores))
        more = np.where(self._scores > mean + 3 * var)[0]
        less = np.where(self._scores < mean - 3 * var)[0]
        print(mean, var)
        print(self._scores)

        # todo: make np.concatenate(less, more, self.scores) work

        indexes = []
        for index in less:
            indexes.append(index)
        for index in more:
            indexes.append(index)
        for index in self._possible_cut:
            indexes.append(index)
        indexes = np.array(indexes)
        return indexes, self._scores[indexes]
