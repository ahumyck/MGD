import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from api_example.learning.learning_lr import cast_arrays
from vcd.analyzer.score.regression_model import RandomForestScoreAnalyzer, load_model


def get_data(filename):
    dataframe = pd.read_excel(filename, index_col=0)
    return cast_arrays(dataframe['frame average speed'])


def rearrange(input_data):
    return np.concatenate([input_data[0], input_data[1:, -1]])


# todo:add ssim
if __name__ == '__main__':
    root = os.getcwd()
    training_data_filename = os.path.join(root, "vcd/resources/MV/data4.xlsx")
    model_filename = os.path.join(root, "vcd/resources/training/models/lr_0.75.model")

    inputs = get_data(training_data_filename)
    data = rearrange(inputs)
    analyzer = RandomForestScoreAnalyzer(inputs, load_model(model_filename))

    indexes, values = analyzer.analyze()

    size = 30

    print(indexes + size - 1, data[indexes + size - 1])

    plt.plot(data)
    plt.plot(indexes + size - 1, data[indexes + size - 1], 'ro')
    plt.savefig('result5.png')
