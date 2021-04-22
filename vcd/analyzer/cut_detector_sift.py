import cv2
import numpy as np

from core.matcher import match_by_bruteforce_fast, sortMatchersByNorm, average_for_matchers
from vcd.analyzer.cut_detector import CutDetector

INVALID_VALUE = -99999


class CutDetectorSIFT(CutDetector):
    def __init__(self, video_name, th=10):
        super().__init__(video_name)
        self.__th = th
        self._sift = cv2.SIFT_create()

    def search_for_slices(self):
        capture = cv2.VideoCapture(self._video_name)  # получаем поток видео
        success, prev_image = capture.read()  # получаем очередное изображени е из видео
        kp_prev, des_prev = self._sift.detectAndCompute(prev_image, None)

        scores = []
        n = 0

        while success:  # до тех пор, пока есть изображения в видео
            success, current_image = capture.read()  # берем следующее изображение
            if success:
                print(n)
                kp_current, des_current = self._sift.detectAndCompute(current_image, None)

                if des_prev is None or des_current is None:
                    kp_prev = kp_current
                    des_prev = des_current
                    scores.append(0)
                    n += 1
                    continue

                # ищем совпадающие ключевые точки
                matchers = match_by_bruteforce_fast(kp_prev, des_prev, kp_current, des_current, th=self.__th)
                if len(matchers) != 0:
                    # сортируем
                    matchers = sortMatchersByNorm(matchers)
                    # берем 5 "лучших" точек
                    matchers = matchers[0:5]
                    # считаем среднее движение кадра
                    scores.append(np.mean(average_for_matchers(matchers, kp_prev, kp_current)))
                else:
                    scores.append(INVALID_VALUE)
                kp_prev = kp_current
                des_prev = des_current
                n += 1

            else:
                break

        capture.release()  # возвращаем ресурсы компьютеру
        return scores
