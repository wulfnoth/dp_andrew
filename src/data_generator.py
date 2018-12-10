import numpy as np
import pandas as pd

num_feature = 2
num_sample = 500


def calculate(value):
    return 2.5*value-0.5


def give_label(feature_vector):
    return 1 if (feature_vector[1] > calculate(feature_vector[0])) else 0


x_values = None

for i in range(num_sample):
    if x_values is None:
        x_values = np.random.rand(num_feature, 1)
    else:
        x_values = np.hstack((x_values, np.random.rand(num_feature, 1)))


y = np.apply_along_axis(give_label, axis=0, arr=x_values).reshape((1, num_sample))

records = np.vstack((x_values, y))

result = pd.DataFrame(records.T, columns=('x1', 'x2', 'y'))
result['y'] = result['y'].astype('int')

result.to_csv("../data/data.csv", index=False)


# plt.show()
