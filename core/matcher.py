from operator import attrgetter

import cv2
import numpy as np


def create_match(distance, trainIdx, queryIdx, imgIdx=0):
    match = cv2.DMatch()  # создаем объект типа DMatch
    match.trainIdx = trainIdx  # индекс обучающего дискриптора
    match.queryIdx = queryIdx  # индекс тестового дискриптора
    match.imgIdx = imgIdx  # индекс обучающего изображения
    match.distance = distance  # используем L2 норму, согласно документации OpenCV
    return match


def match_by_bruteforce_min_norm(kp1, des1, kp2, des2, dx=40, dy=40, th=10):
    """
        Ищем совпадения особых точек считая L2 норму разницы дискрипторов
        Если L2 норма меньше некоторого порогового значения, то считаем что ключевые точки совпадают, иначе нет
        Из всех подходящех совпадений выбираем совпадения с минимальной нормой
    :param kp1: ключевые точки обучающего изображениея
    :param kp2: ключевые точки тестового изображениея
    :param des1: дискрипторы ключевых точек обучающего изображения
    :param des2: дискрипторы ключевых точек тестового изображения
    :param th: порог для L2 нормы
    :param dx: максимальная разность координат точек по оси х
    :param dy: максимальная разность координат точек по оси у
    :return: список matcher'-ов
    """
    possible_matches_array = []
    for trainIdx, d1 in enumerate(des1):
        possible_matches = []
        for queryIdx, d2 in enumerate(des2):
            d = np.linalg.norm(d1 - d2)
            if d <= th:
                keypoint1 = kp1[trainIdx]
                keypoint2 = kp2[queryIdx]
                if np.abs(np.round(keypoint2.pt[1] - keypoint1.pt[1])) <= dx and \
                        np.abs(np.round(keypoint2.pt[0] - keypoint1.pt[0])) <= dy:
                    possible_matches.append(create_match(d, trainIdx, queryIdx, 0))
        if len(possible_matches) > 0:
            possible_matches_array.append(possible_matches)

    return np.array([min(possible_matches, key=attrgetter('distance')) for possible_matches in possible_matches_array],
                    dtype=cv2.DMatch)


def match_by_bruteforce_fast(kp1, des1, kp2, des2, dx=40, dy=40, th=10, limit=15):
    """
        Ищем совпадения особых точек считая L2 норму разницы дискрипторов
        Если L2 норма меньше некоторого порогового значения, то считаем что ключевые точки совпадают, иначе нет
    :param kp1: ключевые точки обучающего изображениея
    :param kp2: ключевые точки тестового изображениея
    :param des1: дискрипторы ключевых точек обучающего изображения
    :param des2: дискрипторы ключевых точек тестового изображения
    :param th: порог для L2 нормы
    :param limit: ограниче сверху для кол-ва matcher'-ов
    :param dx: максимальная разность координат точек по оси х
    :param dy: максимальная разность координат точек по оси у
    :return: список matcher'-ов
    """
    i = 0
    result = []
    for trainIdx, d1 in enumerate(des1):
        for queryIdx, d2 in enumerate(des2):
            d = np.linalg.norm(d1 - d2)
            if d <= th:
                keypoint1 = kp1[trainIdx]
                keypoint2 = kp2[queryIdx]
                if np.abs(np.round(keypoint2.pt[1] - keypoint1.pt[1])) <= dx and \
                        np.abs(np.round(keypoint2.pt[0] - keypoint1.pt[0])) <= dy:
                    result.append(create_match(d, trainIdx, queryIdx, 0))
                    i += 1
                    if i >= limit:
                        return np.array(result, dtype=cv2.DMatch)
                    break
    return np.array(result, dtype=cv2.DMatch)


def drawMatches(img1, kp1, img2, kp2, matches):
    """
        Функция для рисования matches
        Честно повзаимствована из https://github.com/rmislam/PythonSIFT/blob/master/template_matching_demo.py
    """
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    nWidth = w1 + w2
    nHeight = max(h1, h2)
    hdif = int((h2 - h1) / 2)
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

    for i in range(3):
        newimg[hdif:hdif + h1, :w1, i] = img1
        newimg[:h2, w1:w1 + w2, i] = img2

    for m in matches:
        pt1 = (int(kp1[m.trainIdx].pt[0]), int(kp1[m.trainIdx].pt[1] + hdif))
        pt2 = (int(kp2[m.queryIdx].pt[0] + w1), int(kp2[m.queryIdx].pt[1]))
        cv2.line(newimg, pt1, pt2, (255, 0, 0))

    return newimg


def sortMatchersBy(matchers, attribute_name):
    """
        Сортировка совпадений по заданному атрибуту
    :param matchers: список совпадений
    :param attribute_name: имя атрибута
    :return: отсортированный список совпадений по атрибуту
    """
    return sorted(matchers, key=attrgetter(attribute_name))


def sortMatchersByNorm(matchers):
    """
        Сортировка по L2 норме
    :param matchers: список совпадений
    :return: отсортированный список по L2 норме (в возрастающем порядке)
    """
    return sortMatchersBy(matchers, 'distance')


def calculateDistance(point1, point2):
    """
        Вычисление расстояния между точками по теореме пифагора
    :param point1: точка А
    :param point2: точка Б
    :return: расстояние между точками А и Б
    """
    dx = point1.pt[0] - point2.pt[0]
    dy = point1.pt[1] - point2.pt[1]
    return np.sqrt(dx * dx + dy * dy)


def average_for_matchers(matchers, kp_prev, kp_current):
    """
        Функция для подсчета среднего смещения кадра по ключевым точкам
    :param kp_current: ключевые точки с предыдущего изображения
    :param kp_prev: ключевые точки с предыдущего изображения
    :param matchers: лучшия совпадения особых точек с двух изображений
    :return: среднее смещение кадра
    """
    statistics = []
    for matcher in matchers:
        kp1 = kp_prev[matcher.trainIdx]
        kp2 = kp_current[matcher.queryIdx]
        statistics.append(calculateDistance(kp1, kp2))
    return np.mean(statistics)
