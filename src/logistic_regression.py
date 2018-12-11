import numpy as np
import pandas as pd


class LogisticRegression:

    def __init__(self, dim_x):
        self.features_data = None
        self.labels = None
        self.iter = 10
        self.b = np.zeros(shape=(1, 1))
        self.alpha = 0.1
        self.w = np.zeros(shape=(dim_x, 1))

    def set_iter(self, iteration):
        self.iter = iteration

    def transform(self, features_data, labels):
        self.features_data = features_data
        self.labels = labels
        for i in range(self.iter):
            self.lg()

    def lg(self):
        z = np.dot(self.w.T, self.features_data) + self.b
        a = 1/(1+np.exp(-z))
        dz = a-self.labels
        dw = np.sum(self.features_data*dz, axis=1)
        db = np.sum(dz, axis=1)

        self.w -= self.alpha*dw
        self.b -= self.alpha*db

    def print(self):
        print("w is: \n\t", self.w)
        print("b is: \n\t", self.b)


if __name__ == '__main__':
    data = pd.read_csv("../data/data.csv", dtype={'x1': np.float64, 'x2': np.float64, 'y': np.int32})
    m = len(data)
    nd_label = data['y'].values.reshape((1, m))
    nd_data = data.drop(columns='y').values.T

    print(nd_label)
    print(nd_data.shape)
