import time

from core.analyzer.analyzer import FramesSliceSearcherSSIM
from core.builder.builder import VideoBuilder
from core.slicer.spliterator import VideoSlicer, Spliterator
from core.utils import make_template_jpg

if __name__ == '__main__':
    path_to_video = "resources/video/video.mp4"
    output_path = "resources/output/output.mp4"

    frame_size = (600, 800)
    # fps = 60

    spl = Spliterator(path_to_video)  # объект для разбития изображения на фреймы
    print('Cropping video, may take a while...')
    fps = spl.save_frames(frame_size)  # сохраняем фреймы в разрешении 600х800, для ускорения работы алгоритма

    slicer = VideoSlicer(path_to_video)  # создаем объект, для разбиения видео на склейки

    r = slicer.slice_video(0, 3) + slicer.slice_video(9, 7)  # + slicer.slice_video(5, 4)

    # создаем объект для записи видео
    builder = VideoBuilder(output_path, make_template_jpg(path_to_video), fps, frame_size)
    builder.compile_and_save_video(r)  # сохраняем нашу склейку

    # создаем объект для поиска склеек на видео
    slice_searcher = FramesSliceSearcherSSIM(path_to_video, r)
    start = time.time()  # таймер
    scores = slice_searcher.search_for_slices()  # получение результатов
    end = time.time()
    print('it took me {}'.format(end - start))
    i, s = slice_searcher.analyze_scores()
    print(i, s)
