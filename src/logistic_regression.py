import numpy as np


class LogisticRegression:

    def __init__(self, dim_x):
        self.b = np.zeros(shape=(1, 1))
        self.alpha = 0.1
        self.w = np.zeros(shape=(dim_x, 1))
        # self.w = np.random.rand(dim_x, 1)

    def lg(self, x, y):
        z = np.dot(self.w.T, x) + self.b
        a = 1/(1+np.exp(-z))
        dz = a-y
        dw = np.sum(x*dz, axis=1)
        db = np.sum(dz, axis=1)

        self.w -= self.alpha*dw
        self.b -= self.alpha*db

    def print(self):
        print("w is: \n\t", self.w)
        print("b is: \n\t", self.b)


if __name__ == '__main__':
    x = np.arange(1, 10).reshape((3, 3)).T
    print(x)
    print(np.sum(x, axis=0, keepdims=True))
    # dz = np.array([3, 4, 6]).reshape(1, 3)
    # # dw = np.multiply(x, dz)
    # dw = x*dz
    # print(x)
    # print(dw)
    # x = np.array([1, 2, 3, 4, 5, 6])
    # x = np.array([3, 4]).reshape(2, 1)
    # y = np.array([1]).reshape(1, 1)
    # lg = LogisticRegression(2)
    # lg.lg(x, y)
    # lg.print()
