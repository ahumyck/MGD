import cv2


# Класс, для создания видео из видеоряда, полученным классом VideoSlier
# Пишет warning, но вроде работает
class VideoBuilder:
    def __init__(self, output_video_name, template_images_name, fps=60, frame_size=(1920, 1080)):
        self.__output_name = output_video_name
        self.__template = template_images_name
        self.__fps = fps
        self.__frame_size = frame_size

    def compile_and_save_video(self, indexes):
        codec = cv2.VideoWriter_fourcc(*'MP42')
        out = cv2.VideoWriter(self.__output_name, codec, self.__fps, self.__frame_size)
        for index in indexes:
            filename = self.__template.format(index)
            img = cv2.imread(filename)
            out.write(img)
        out.release()
