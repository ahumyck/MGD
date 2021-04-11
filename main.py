import time

from vcd.analyzer.cut_detector_sift import CutDetectorSIFT

if __name__ == '__main__':
    path_to_video = "vcd/resources/output/output.mp4"

    # создаем объект для поиска склеек на видео
    slice_searcher = CutDetectorSIFT(path_to_video)
    start = time.time()  # таймер
    scores = slice_searcher.search_for_slices()  # получение результатов
    end = time.time()
    print('it took me {}'.format(end - start))
    i, s = slice_searcher.analyze_scores()
    print(i, s)
