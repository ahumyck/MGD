import os
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt

from vcd.analyzer.score.score_analyzer import EmpiricalRuleScoreAnalyzer, more_op, less_op
from vcd.analyzer.score_collector_ssim import ScoreCollectorSSIM


def get_videos(folder):
    return [f for f in listdir(folder) if isfile(join(folder, f))]


def upper_th(mean, sigma):
    return mean + 3 * sigma


def lower_th(mean, sigma):
    return mean - 3 * sigma


def graphics_simple_case(inputs, fig_name, is_upper=True):
    if is_upper:
        operation = more_op
        th_calculator = upper_th
    else:
        operation = less_op
        th_calculator = lower_th

    score_analyzer = EmpiricalRuleScoreAnalyzer(inputs, 25, [operation])
    ind, values = score_analyzer.analyze()
    mean = score_analyzer.get_mean()
    sigma = score_analyzer.get_sigma()

    print(score_analyzer.cast_frame_index_to_time(ind))

    const = th_calculator(mean, sigma)

    x = [0, len(inputs) - 1]
    y = [const, const]

    plt.plot(inputs)
    plt.plot(ind, values, 'ro')
    plt.plot(x, y, color='red')
    plt.ylabel('Мера близости')
    plt.savefig(fig_name)

    plt.cla()
    plt.clf()


def graphics_sift_case(pandas_files):
    pass


if __name__ == '__main__':
    path_to_videos = 'C:/Users/ahumyck/PycharmProjects/diplom/vcd/resources/CCTV'
    videos = get_videos(path_to_videos)

    for video_path in videos:
        full_video_path = os.path.join(path_to_videos, video_path)
        print('current video =', full_video_path)
        video_template = os.path.splitext(video_path)[0]
        fig_template = os.path.join(path_to_videos, '{}_' + video_template + '.png')
        ssim_collector = ScoreCollectorSSIM(full_video_path)
        ssim_scores = ssim_collector.collect()
        graphics_simple_case(ssim_scores[:-10], fig_template.format('SSIM'), is_upper=False)
