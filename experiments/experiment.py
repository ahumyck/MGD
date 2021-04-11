import cv2
import matplotlib.pyplot as plt

from vcd.core.handler import SaltAndPepperHandler, MoveByHandler
from vcd.core.matcher import match_by_bruteforce_min_norm, match_by_bruteforce_fast, drawMatches
from vcd.core.utils import float_range

WHITE = 255
GRAY = 127
BLACK = 0


def abstract_experiment(img1, img2, output_image_name, algorithm, th):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matches = algorithm(kp1, des1, kp2, des2, th=th)  # ищем совпадения
    img3 = drawMatches(img1, kp1, img2, kp2, matches)  # рисуем совпадения
    plt.imshow(img3)
    plt.savefig(output_image_name)


def make_experiment_fast(img1, img2, output_image_name, th):
    abstract_experiment(img1, img2, output_image_name, match_by_bruteforce_fast, th)


def make_experiment_correct(img1, img2, output_image_name, th):
    abstract_experiment(img1, img2, output_image_name, match_by_bruteforce_min_norm, th)


if __name__ == '__main__':
    train_image = cv2.imread('core/box.png', cv2.IMREAD_GRAYSCALE)  # train image

    move_by_handler = MoveByHandler(train_image)

    print('move experiments')
    for x in range(0, 50, 10):
        for y in range(0, 50, 10):
            make_experiment_fast(train_image, move_by_handler.handle(x, y, BLACK),
                                 f'core/move_by/x = {x}, y = {y}.png', th=10)

    noise_handler = SaltAndPepperHandler(train_image)

    print('s&p experiments')

    for sp in float_range(0.1, 0.7, 0.1):
        for amount in float_range(0.05, 0.4, 0.05):
            make_experiment_fast(train_image, noise_handler.handle(sp, amount),
                                 f'core/salt and pepper/probability = {round(sp, 2)}, '
                                 f'amount = {round(amount, 4)}.png', th=50)

    print('done')
