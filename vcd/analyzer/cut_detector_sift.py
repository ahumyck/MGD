import time

import cv2

from core.matcher import match_by_bruteforce_fast, sortMatchersByNorm, average_for_matchers
from vcd.analyzer.cut_detector import CutDetector


class CutDetectorSIFT(CutDetector):
    def __init__(self, video_name, matches_limit=15, global_th=20, local_th=75):
        super().__init__(video_name)
        self.__global_th = global_th
        self.__local_th = local_th
        self.__matches_limit = matches_limit
        self._sift = cv2.SIFT_create()

    def search_for_slices(self):
        capture = cv2.VideoCapture(self._video_name)  # получаем поток видео
        success, prev_image = capture.read()  # получаем очередное изображение из видео

        if not success:
            raise Exception(f'fail on loading video {self._video_name}')

        kp_prev, des_prev = self._sift.detectAndCompute(prev_image, None)

        scores = []
        n = 0

        while success:  # до тех пор, пока есть изображения в видео
            success, current_image = capture.read()  # берем следующее изображение
            if success:
                kp_current, des_current = self._sift.detectAndCompute(current_image, None)

                if kp_current is None or des_current is None:
                    capture.release()  # возвращаем ресурсы компьютеру
                    return scores

                start = time.time()
                current_local_th = self.__local_th
                # ищем совпадающие ключевые точки
                while True:
                    matchers = match_by_bruteforce_fast(kp_prev, des_prev, kp_current, des_current,
                                                        global_th=self.__global_th,
                                                        local_th=current_local_th,
                                                        limit=self.__matches_limit)
                    if len(matchers) == 0:
                        current_local_th += 50
                    else:
                        end = time.time()
                        print(
                            f'n = {n}, current th = {current_local_th}, len(matchers) = {len(matchers)}, '
                            f'len(kp_current) = {len(kp_current)}, time for match = {end - start}'
                        )
                        break
                # сортируем
                matchers = sortMatchersByNorm(matchers)

                matchers = matchers[:5]

                # считаем среднее движение кадра
                scores.append(average_for_matchers(matchers, kp_prev, kp_current))
                kp_prev = kp_current
                des_prev = des_current
                n += 1

            else:
                break

        capture.release()  # возвращаем ресурсы компьютеру
        return scores
