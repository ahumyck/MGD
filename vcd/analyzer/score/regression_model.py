import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_mode(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    file = 'filename.model'
    x = np.random.rand(1000, 30)
    y = np.where(np.random.rand(1000, 1) > 0.5, 1, 0)

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.5, random_state=42)

    classes = ['Склейка', 'Не склейка']
    lr = LogisticRegression()
    lr.fit(xtrain, ytrain)

    predict = lr.predict(xtest)
    print(confusion_matrix(ytest, predict))
    print(classification_report(ytest, predict, target_names=classes))

    save_mode(lr, file)

    loaded = load_model(file)
    predict = loaded.predict(xtest)
    print(confusion_matrix(ytest, predict))
    print(classification_report(ytest, predict, target_names=classes))
