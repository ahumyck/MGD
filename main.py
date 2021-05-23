import numpy as np

from vcd.analyzer.score.regression_model import convert_scores, RegressionModelScoreAnalyzer, load_model
from vcd.analyzer.score.score_analyzer import EmpiricalRuleScoreAnalyzer, less_op
from vcd.analyzer.score_collector_sift import ScoreCollectorSIFT
from vcd.analyzer.score_collector_ssim import ScoreCollectorSSIM

if __name__ == '__main__':
    video_path = input("Укажите полный путь к видео:")
    algorithm_type = input("Вариант анализа видео(SIFT или SSIM):")

    if algorithm_type is "SIFT":
        feature_vector_size = 30
        model_path = input("Укажите путь к модели:")
        collector_sift = ScoreCollectorSIFT(video_path, local_th=75)
        scores = np.array(collector_sift.collect())
        X = convert_scores(scores, feature_vector_size)
        analyzer = RegressionModelScoreAnalyzer(scores, load_model(model_path))
        indexes, values = analyzer.analyze()
        print("Обнаруженные подозрительные места")
        print(indexes + 1, scores[indexes + 1])
    else:
        collector_ssim = ScoreCollectorSSIM(video_path)
        scores = np.array(collector_ssim.collect())
        analyzer = EmpiricalRuleScoreAnalyzer(scores, less_op)
        indexes, value = analyzer.analyze()
        print("Обнаруженные подозрительные места")
        print(indexes + 1, scores[indexes + 1])
