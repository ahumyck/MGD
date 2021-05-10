from vcd.spliterator import Spliterator

if __name__ == '__main__':
    path_to_video = "/training/video\\result.mp4"

    frame_size = (600, 800)
    # fps = 60

    spl = Spliterator(path_to_video)  # объект для разбития изображения на фреймы
    print('Cropping video, may take a while...')
    spl.save_frames(frame_size)  # сохраняем фреймы в разрешении 600х800, для ускорения работы алгоритма
