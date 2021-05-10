import abc


class ScoreCollector:
    def __init__(self, video_name):
        self._video_name = video_name

    @abc.abstractmethod
    def collect(self):
        """
            Метод для получения оценок из видео
        """
        return




