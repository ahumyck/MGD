import os

import cv2


def get_frames_and_length_of_video(video_name):
    """
        Получение кол-во кадров видео и его длинны в секундах
    :param video_name: Название видео
    :return: кол-во кадров в видео, длина видео
    """
    capture = cv2.VideoCapture(video_name)
    fps = capture.get(cv2.CAP_PROP_FPS)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return count, float(count) / float(fps)


def float_range(start, stop, step):
    """
        Аналог range для float чисел
    """
    if stop < start:
        start, stop = stop, start
    elif stop == start:
        return []

    for i in range(int((stop - start) / step)):
        yield start + i * step


def make_template(filename, ext):
    return os.path.splitext(filename)[0] + "{}" + ext


def make_template_jpg(filename):
    return make_template(filename, ".jpg")
