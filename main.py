import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from vcd.analyzer.score.regression_model import save_mode


def cast_arrays(arrays):
    def cast_str_to_array_of_numbers(arr: str):
        s = arr[1:-1]
        elements = s.split(",")
        res = []
        for element in elements:
            res.append(float(element.strip()))
        return np.array(res)

    result = []
    length = []
    for array in arrays:
        arr = cast_str_to_array_of_numbers(array)
        length.append(len(arr))
        result.append(arr)

    index = min(length)
    for i in range(len(result)):
        arr = result[i]
        result[i] = arr[-index:]

    return np.array(result)


def get_training_data(filename):
    dataframe = pd.read_excel(filename, index_col=0)
    return cast_arrays(dataframe['v(t) series']), dataframe['y series'].to_numpy()


def learning(training_data, test_size, model_name=None, roc_auc_curve_name=None):
    x, y = training_data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    # classes = ['Не склейка', 'склейка']
    lr = LogisticRegression()
    lr.fit(x_train, y_train)

    # predict = lr.predict(x_test)

    # print(confusion_matrix(y_test, predict))
    # print(classification_report(y_test, predict, target_names=classes))
    # print(roc_auc_score(y, lr.predict_proba(x)[:, 1]))

    if model_name is not None:
        save_mode(lr, model_name)

    if roc_auc_curve_name is not None:
        metrics.plot_roc_curve(lr, x_test, y_test)
        plt.savefig(roc_auc_curve_name)


if __name__ == '__main__':
    root = os.getcwd()
    training_data_filename = os.path.join(root, "vcd/resources/data/data_ready.xlsx")
    model_template_name = os.path.join(root, "vcd/resources/models/lr_{}.model")
    roc_auc_curve_template_name = os.path.join(root, "vcd/resources/result/roc_auc_{}.png")

    x, y = get_training_data(training_data_filename)

    epsilon = 1e-9

    test_sizes = np.arange(0.01, 0.9 + epsilon, 0.01)  # np.arange(0, 1.1, 0.5) => [0, 0.5, 1.0]
    for test_size in test_sizes:
        print(test_size)
        model_name = model_template_name.format(format(test_size, '.2f'))
        roc_auc_name = roc_auc_curve_template_name.format(format(test_size, '.2f'))

        learning((x, y), test_size, model_name, roc_auc_name)

    plt.close('all')
