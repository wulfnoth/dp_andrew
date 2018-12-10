import matplotlib.pyplot as plot
import numpy as np
import pandas as pd

fig = plot.figure(1, figsize=(6, 4), dpi=100)

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

data = pd.read_csv("../data/data.csv", dtype={'x1': np.float64, 'x2': np.float64, 'y': np.int32})

colors = ['red', 'blue'] # red is negative, blue is positive
for index, row in data.iterrows():
    ax.scatter(row['x1'], row['x2'], color=colors[int(row['y'])])

plot.show()

