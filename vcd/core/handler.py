import numpy as np


class ImageHandler:
    def __init__(self, img):
        self.img = img

    def handle(self, *args):
        pass


class MoveByHandler(ImageHandler):
    def __init__(self, img):
        super().__init__(img)

    def handle(self, by_x, by_y, color):
        """
            Перемещаем исходное изображение by_x пикселей по горизонтали
            и by_y пикселей по вертикали
            "освободившееся" пространство заполняем указанным цветом
        """
        h, w = self.img.shape  # размер изображения
        copy = np.copy(self.img)  # создаем копию
        copy[by_x:, by_y:] = copy[:h - by_x, :w - by_y]  # сдвигаем
        # освободившееся место заполняем указанным цветом
        copy[:by_x, :] = color
        copy[:, :by_y] = color
        return copy


class SaltAndPepperHandler(ImageHandler):
    def __init__(self, img):
        super().__init__(img)

    def handle(self, sp, amount):
        """
            Функция для зашумления изображения способов
            "соль и перец"
        """
        out = np.copy(self.img)
        # Salt mode
        num_salt = np.ceil(amount * self.img.size * sp)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in self.img.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * self.img.size * (1. - sp))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in self.img.shape]
        out[coords] = 0
        return out
