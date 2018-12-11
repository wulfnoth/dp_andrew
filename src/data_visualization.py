import matplotlib.pyplot as plot
import numpy as np
import pandas as pd


def pt(data, label, line):
    plot.figure(1, figsize=(6, 4), dpi=100)

    ax = plot.subplot(111)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(0, 1.2)
    ax.set_xticks([0, 1])
    ax.set_yticks([1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')

    # data = pd.read_csv("../data/data.csv", dtype={'x1': np.float64, 'x2': np.float64, 'y': np.int32})
    # nd_data = data.values

    colors = ['red', 'blue']  # red is negative, blue is positive
    shape = data.shape
    x_lim = np.linspace(-0.2, 1.2)
    print(line(x_lim))
    ax.plot(x_lim, line(x_lim))
    for i in range(shape[1]):
        column = data[:, i]
        # print(column[0])
        ax.scatter(column[0], column[1], color=colors[int(label[0, i])])
    # for index, row in data.iterrows():
    #     ax.scatter(row['x1'], row['x2'], color=colors[int(row['y'])])

    plot.show()

