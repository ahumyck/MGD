import cv2
import matplotlib.pyplot as plt

import matcher


def abstract_experiment(img1, img2, output_image_name, algorithm, th):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matches = algorithm(kp1, des1, kp2, des2, th=th)  # ищем совпадения
    img3 = matcher.drawMatches(img1, kp1, img2, kp2, matches)  # рисуем совпадения
    plt.imshow(img3)
    plt.savefig(output_image_name)


def make_experiment_fast(img1, img2, output_image_name, th):
    abstract_experiment(img1, img2, output_image_name, matcher.match_by_bruteforce_fast, th)


def make_experiment_correct(img1, img2, output_image_name, th):
    abstract_experiment(img1, img2, output_image_name, matcher.match_by_bruteforce_min_norm, th)
