import cv2

from core.utils import make_template_jpg, get_frames_and_length_of_video


# Класс для получения фреймов из видео
class Spliterator:  # создание объекта для разбиения видео на изображения
    def __init__(self, video_name):
        self.__video_name = video_name
        self.__template = make_template_jpg(video_name)  # создаем шаблон для сохранения изображений

    def save_frames(self, new_size=None):
        capture = cv2.VideoCapture(self.__video_name)  # получаем поток видео
        count = 0
        while True:  # до тех пор, пока есть изображения в видео
            success, image = capture.read()  # берем следующее изображение
            if success:
                if new_size is not None:  # если надо, меняем размер изображения
                    image = cv2.resize(image, new_size)
                cv2.imwrite(self.__template.format(count), image)  # сохраняем картинку на компьютере
                count += 1
            else:
                break
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()  # возвращаем ресурсы компьютеру
        return fps  # возвращаем кол-во кадров в секунду


# Класс для удобной склейки фреймов из видеоряда
class VideoSlicer:
    def __init__(self, video_name):  # создание объекта для разбиения видео подвидео
        self.__video_name = video_name
        self.__template = make_template_jpg(video_name)  # создаем шаблон для загрузки изображений
        self.__frames, self.__video_length = get_frames_and_length_of_video(self.__video_name)

    # получение подвидео
    def slice_video(self, start, end):
        if start is None and end is None:  # если не указано ни начало, ни конец, то возвращаем все видео
            return list(range(0, self.__frames, 1))
        s = int((start / self.__video_length) * self.__frames)  # переводим начало (в секундах) в номер кадра
        e = int((end / self.__video_length) * self.__frames)  # переводим конец (в секундах) в номер кадра

        # если начало больше чем конец, возвращаем индексы кадров в обратном порядке, чтобы видео шло в обратную сторону
        if s > e:
            # проверка, чтобы мы не вышли за границы границы длины видео
            s = self.__frames - 1 if s >= self.__frames else s - 1
            e = -1 if e < 0 else e - 1
            return list(range(s, e, -1))
        elif s == e:  # если начало и конец совпадают, возвращаем один кадр
            # проверка, чтобы мы не вышли за границы границы длины видео
            s = self.__frames - 1 if s >= self.__frames else s
            s = 0 if s < 0 else s
            return [s]
        else:
            # проверка, чтобы мы не вышли за границы границы длины видео
            s = 0 if s < 0 else s
            e = self.__frames if e >= self.__frames else e
            return list(range(s, e, 1))
