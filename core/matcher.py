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


def match_by_bruteforce_min_norm(kp1, des1, kp2, des2, dx=50, dy=50, th=50):
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


def search_for_matches(kp1, des1, kp2, des2, dx=40, dy=40, global_th=10, local_th=50, limit=15):
    """
        Ищем совпадения особых точек считая L2 норму разницы дискрипторов
        Если L2 норма меньше некоторого порогового значения, то считаем что ключевые точки совпадают, иначе нет
    :param kp1: ключевые точки обучающего изображениея
    :param kp2: ключевые точки тестового изображениея
    :param des1: дискрипторы ключевых точек обучающего изображения
    :param des2: дискрипторы ключевых точек тестового изображения
    :param local_th: порог для L2 нормы, в случае если изменения произошли в небольшом квадранте
    :param global_th: порог для L2 нормы в рамках всего изображения
    :param limit: ограниче сверху для кол-ва matcher'-ов
    :param dx: максимальная разность координат точек по оси х
    :param dy: максимальная разность координат точек по оси у
    :return: список matcher'-ов
    """
    counters = 0
    result = []
    for trainIdx, d1 in enumerate(des1):
        for queryIdx, d2 in enumerate(des2):
            d = np.linalg.norm(d1 - d2)
            if d <= global_th:
                result.append(create_match(d, trainIdx, queryIdx, 0))
                counters += 1
                break
            elif d <= local_th:
                keypoint1 = kp1[trainIdx]
                keypoint2 = kp2[queryIdx]
                if np.abs(keypoint2.pt[1] - keypoint1.pt[1]) <= dx and np.abs(keypoint2.pt[0] - keypoint1.pt[0]) <= dy:
                    result.append(create_match(d, trainIdx, queryIdx, 0))
                    counters += 1
                    break
        if counters >= limit:
            return np.array(result, dtype=cv2.DMatch)
    return np.array(result, dtype=cv2.DMatch)


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


def average_for_matchers(matchers, kpf, kps, how_many=5):
    """
        Функция для подсчета среднего смещения кадра по ключевым точкам
    :param kps: ключевые точки со следующего изображения
    :param kpf: ключевые точки с предыдущего изображения
    :param matchers: лучшия совпадения особых точек с двух изображений
    :param how_many: кол-во точек, которое стоит использовать для сбора статистики
    :return: среднее смещение кадра
    """
    statistics = []
    for matcher in matchers:
        kp1 = kpf[matcher.trainIdx]
        kp2 = kps[matcher.queryIdx]
        statistics.append(calculateDistance(kp1, kp2))
    sorted_stats = sorted(statistics)
    return np.mean(sorted_stats[:how_many])
