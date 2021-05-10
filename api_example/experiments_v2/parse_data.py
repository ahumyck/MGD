import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from api_example.learning.learning import cast_arrays
from vcd.analyzer.score.regression_model import RegressionModelScoreAnalyzer, load_model


def get_data(filename):
    dataframe = pd.read_excel(filename, index_col=0)
    return cast_arrays(dataframe['frame average speed'])


def rearrange(input_data):
    N, M = input_data.shape
    result = []

    for v in input_data[0]:
        result.append(v)

    for i in range(1, N):
        result.append(input_data[i][-1])

    return np.array(result)


def rearrange_v2(input_data):
    return np.array([input_data[i][-1] for i in range(len(input_data))])


if __name__ == '__main__':
    root = os.getcwd()
    training_data_filename = os.path.join(root, "vcd/resources/MV/data4.xlsx")
    model_filename = os.path.join(root, "vcd/resources/training/models/lr_0.75.model")

    inputs = get_data(training_data_filename)
    data = rearrange(inputs)
    analyzer = RegressionModelScoreAnalyzer(inputs, 30, 25, load_model(model_filename))

    indexes, values = analyzer.analyze()
    re_arranged = rearrange_v2(values)

    print(indexes + 2, data[indexes])

    plt.plot(data)
    plt.plot(indexes, data[indexes], 'ro')
    plt.savefig('result4.png')
