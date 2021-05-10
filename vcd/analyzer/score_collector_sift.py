import cv2

from core.matcher import search_for_matches, sortMatchersByNorm, average_for_matchers
from vcd.analyzer.score_collector import ScoreCollector


class ScoreCollectorSIFT(ScoreCollector):
    def __init__(self, video_name, matches_limit=15, global_th=20, local_th=75):
        super().__init__(video_name)
        self.__global_th = global_th
        self.__local_th = local_th
        self.__matches_limit = matches_limit
        self._sift = cv2.SIFT_create(nfeatures=500)

    def collect(self):
        capture = cv2.VideoCapture(self._video_name)  # получаем поток видео
        success, prev_image = capture.read()  # получаем очередное изображение из видео

        kp_prev, des_prev = self._sift.detectAndCompute(prev_image, None)
        scores = []

        n = 0

        while success:  # до тех пор, пока есть изображения в видео
            print(n)
            success, current_image = capture.read()  # берем следующее изображение
            if success:
                kp_current, des_current = self._sift.detectAndCompute(current_image, None)

                if kp_current is None or des_current is None:
                    capture.release()
                    return scores

                current_local_th = self.__local_th
                allMatches = []
                while len(allMatches) == 0:
                    matches = search_for_matches(kp_prev, des_prev, kp_current, des_current,
                                                 global_th=self.__global_th,
                                                 local_th=current_local_th,
                                                 limit=self.__matches_limit)
                    allMatches.extend(matches)
                    current_local_th += 50

                allMatches = sortMatchersByNorm(allMatches)
                scores.append(average_for_matchers(allMatches[:5], kp_prev, kp_current))
                kp_prev = kp_current
                des_prev = des_current

            n += 1

        capture.release()  # возвращаем ресурсы компьютеру
        return scores
