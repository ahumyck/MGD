import cv2
import matplotlib.pyplot as plt
import numpy as np

import matcher

WHITE = 255
GRAY = 127
BLACK = 0


def move_image_by(img, by_x, by_y, color):
    """
        Перемещаем исходное изображение by_x пикселей по горизонтали
        и by_y пикселей по вертикали
        "освободившееся" пространство заполняем указанным цветом
    """
    h, w = img.shape  # размер изображения
    copy = np.copy(img)  # создаем копию
    copy[by_x:, by_y:] = copy[:h - x, :w - y]  # сдвигаем
    # освободившееся место заполняем указанным цветом
    copy[:by_x, :] = color
    copy[:, :by_y] = color
    return copy


def salt_and_pepper(image, s_vs_p=0.5, amount=0.004):
    """
        Функция для зашумления изображения способов
        "соль и перец"
    """
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    out[coords] = 0
    return out


def make_move_experiment(im1, im2, by_x, by_y):
    """
        Эксперимент для нахождения
        одинаковых ключевых точек у оригинального и сдвинутого изображеия
    """
    moved_im2 = move_image_by(im2, by_x, by_y, BLACK)  # сдвигаем изображение

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(moved_im2, None)

    matches = matcher.match_by_bruteforce_correct(kp1, des1, kp2, des2, x, y)  # ищем совпадения
    img3 = matcher.drawMatches(im1, kp1, moved_im2, kp2, matches)  # рисуем совпадения
    plt.imshow(img3)
    plt.savefig(f'move_by/x = {by_x}, y = {by_y}.png')


def salt_and_pepper_experiment(im1, im2, sp, amount):
    """
        Эксперимент для нахождения
        одинаковых ключевых точек у оригинального и зашумленного изображеия
    """
    sp_im2 = salt_and_pepper(im2, sp, amount)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(sp_im2, None)

    matches = matcher.match_by_bruteforce_correct(kp1, des1, kp2, des2)
    img3 = matcher.drawMatches(im1, kp1, sp_im2, kp2, matches)
    plt.imshow(img3)
    plt.savefig(f'salt and pepper/probability = {round(sp, 2)}, amount = {round(amount, 4)}.png')


def salt_then_move_experiment(im1, im2, by_x, by_y, sp, amount):
    """
        Эксперимент для нахождения
        одинаковых ключевых точек у оригинального и зашумленного и сдвинутого изображеия
    """
    sp_move_im2 = move_image_by(salt_and_pepper(im2, sp, amount), by_x, by_y, BLACK)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(sp_move_im2, None)

    matches = matcher.match_by_bruteforce_correct(kp1, des1, kp2, des2)
    img3 = matcher.drawMatches(im1, kp1, sp_move_im2, kp2, matches)
    plt.imshow(img3)
    plt.savefig(f'move and salt/x = {by_x}, y = {by_y}, probability = {round(sp, 2)}, amount = {round(amount, 4)}.png')


def float_range(start, stop, step):
    """
        Аналог range для float чисел
    """
    float_r = []
    while start < stop:
        float_r.append(start)
        start += step
    return float_r


if __name__ == '__main__':
    img1 = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)  # train image
    img2 = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)  # test image

    print('move experiments')
    for x in range(0, 50, 10):
        for y in range(0, 50, 10):
            make_move_experiment(img1, img2, x, y)

    print('s&p experiments')
    for s_vs_p in float_range(0.1, 0.7, 0.1):
        for amount in float_range(0.01, 0.7, 0.1):
            salt_and_pepper_experiment(img1, img2, s_vs_p, amount)

    print('moved and salted')
    for x in range(0, 50, 10):
        for y in range(0, 50, 10):
            for s_vs_p in float_range(0.3, 0.7, 0.1):
                for amount in float_range(0.3, 0.7, 0.1):
                    salt_then_move_experiment(img1, img2, x, y, s_vs_p, amount)

    print('done')
