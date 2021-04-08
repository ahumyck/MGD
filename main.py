import cv2

import experiment
from handler import SaltAndPepperHandler, MoveByHandler

WHITE = 255
GRAY = 127
BLACK = 0


def float_range(start, stop, step):
    """
        Аналог range для float чисел
    """
    if stop < start:
        start, stop = stop, start

    for i in range(int(((stop - start) / step))):
        yield start + i * step


if __name__ == '__main__':
    img1 = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)  # train image
    img2 = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)  # test image

    move_by_handler = MoveByHandler(img2)

    print('move experiments')
    for x in range(0, 50, 10):
        for y in range(0, 50, 10):
            experiment.make_experiment_fast(img1, move_by_handler.handle(x, y, BLACK),
                                            f'move_by/x = {x}, y = {y}.png', th=10)

    noise_handler = SaltAndPepperHandler(img2)

    print('s&p experiments')

    for sp in float_range(0.1, 0.7, 0.1):
        for amount in float_range(0.05, 0.4, 0.05):
            experiment.make_experiment_fast(img1, noise_handler.handle(sp, amount),
                                            f'salt and pepper/probability = {round(sp, 2)}, '
                                            f'amount = {round(amount, 4)}.png', th=50)

    print('done')
