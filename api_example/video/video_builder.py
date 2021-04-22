from core.utils import make_template_jpg
from vcd.builder import VideoBuilder
from vcd.spliterator import VideoSlicer

if __name__ == '__main__':
    path_to_video = "vcd/resources/video/video1.mp4"
    output_path = "vcd/resources/output/output1.mp4"

    frame_size = (600, 800)
    fps = 60

    slicer = VideoSlicer(path_to_video)  # создаем объект, для разбиения видео на склейки

    r = slicer.slice_video()

    # создаем объект для записи видео
    builder = VideoBuilder(output_path, make_template_jpg(path_to_video), fps, frame_size)
    builder.compile_and_save_video(r)  # сохраняем нашу склейку
