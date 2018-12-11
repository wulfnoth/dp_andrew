import numpy as np
import pandas as pd
import data_visualization as dv


class LogisticRegression:

    def __init__(self, dim_x):
        self.features_data = None
        self.labels = None
        self.iter = 100
        self.b = np.zeros(shape=(1, 1))
        self.alpha = 0.01
        self.w = np.zeros(shape=(dim_x, 1))
        self.m = 0

    def set_iter(self, iteration):
        self.iter = iteration

    def fit(self, features_data, labels):
        _, self.m = features_data.shape
        self.features_data = features_data
        self.labels = labels
        for i in range(self.iter):
            self.lg()

    def lg(self):
        z = np.dot(self.w.T, self.features_data) + self.b
        a = 1 / (1 + np.exp(-z))
        dz = a - self.labels
        # print(self.features_data.shape)
        dw = np.sum(self.features_data * dz, axis=1, keepdims=True) / self.m
        # print(dw)
        db = np.sum(dz, axis=1) / self.m

        self.w -= self.alpha * dw
        self.b -= self.alpha * db

    def print(self):
        print("w is: \n\t", self.w / self.w[1, 0])
        print("b is: \n\t", self.b / self.w[1, 0])


global wr
global br


def cal(x):
    return wr * x + br


if __name__ == '__main__':
    # data = np.array([[1, 2, 3], [4, 5, 6]]).T.reshape(3, 2)
    # z = np.array([10, 100]).reshape((1, 2))
    # print(np.sum(data*z, axis=1, keepdims=True)/2)
    # data = pd.read_csv("../data/test_data.csv", dtype={'x1': np.float64, 'x2': np.float64, 'y': np.int32})
    data = pd.read_csv("../data/data.csv", dtype={'x1': np.float64, 'x2': np.float64, 'y': np.int32})
    m = len(data)
    nd_label = data['y'].values.reshape((1, m))
    nd_data = data.drop(columns='y').values.T

    no_features, _ = nd_data.shape
    lg = LogisticRegression(no_features)
    lg.set_iter(10000)
    lg.fit(nd_data, nd_label)

    ww = lg.w

    wr = -ww[0, 0] / ww[1, 0]
    br = -lg.b[0, 0] / ww[1, 0]
    print(wr, br)

    dv.pt(nd_data, nd_label, cal)

    lg.print()
