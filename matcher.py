import cv2
import numpy as np


def create_match(distance, trainIdx, queryIdx, imgIdx=0):
    match = cv2.DMatch()  # создаем объект типа DMatch
    match.trainIdx = trainIdx  # индекс обучающего дискриптора
    match.queryIdx = queryIdx  # индекс тестового дискриптора
    match.imgIdx = imgIdx  # индекс обучающего изображения
    match.distance = distance  # используем L2 норму, согласно документации OpenCV
    return match


def match_by_bruteforce_correct(kp1, des1, kp2, des2, x=0, y=0, limit=25):
    """
        Ищем совпадения особых точек основываясь на знании о их месторасположении
        Сначала мы проверяем, что сравниваемые точки сдвинута одна относительно другой на заданную величину
        После чего считаем для совпадения L2 норму от разности дискрипторов
    :param kp1: ключевые точки обучающего изображениея
    :param des1: дискрипторы ключевых точек обучающего изображения
    :param kp2: ключевые точки тестового изображениея
    :param des2: дискрипторы ключевых точек тестового изображения
    :param x: Смещение по оси х. Значение х=0 используется в эксперементах с шумами, когда мы не сдвигаем изображение
    :param y: Смещение по оси у. Значение х=0 используется в эксперементах с шумами, когда мы не сдвигаем изображение
    :param limit: ограниче сверху для кол-ва matcher'-ов
    :return: список matcher'-ов
    """
    result = []
    i = 0
    for trainIdx, d1 in enumerate(des1):
        for queryIdx, d2 in enumerate(des2):
            keypoint1 = kp1[trainIdx]
            keypoint2 = kp2[queryIdx]
            if np.round(keypoint2.pt[1] - keypoint1.pt[1]) == x and \
                    np.round(keypoint2.pt[0] - keypoint1.pt[0]) == y:
                result.append(create_match(np.linalg.norm(d1 - d2), trainIdx, queryIdx, 0))
                i += 1
                if i >= limit:
                    return np.array(result, dtype=cv2.DMatch)
                break
    return np.array(result, dtype=cv2.DMatch)


def match_by_bruteforce_fast(des1, des2, th=10, limit=25):
    """
        Ищем совпадения особых точек считая L2 норму разницы дискрипторов
        Если L2 норма меньше некоторого порогового значения, то считаем что ключевые точки совпадают, иначе нет
    :param des1: дискрипторы ключевых точек обучающего изображения
    :param des2: дискрипторы ключевых точек тестового изображения
    :param th: порог для L2 нормы
    :param limit: ограниче сверху для кол-ва matcher'-ов
    :return: список matcher'-ов
    """
    i = 0
    result = []
    for trainIdx, d1 in enumerate(des1):
        for queryIdx, d2 in enumerate(des2):
            d = np.linalg.norm(d1 - d2)
            if d <= th:
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
