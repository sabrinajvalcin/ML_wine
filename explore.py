import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

red_df = pd.read_csv('winequality-red.csv', header = 0, sep= ';')
red_df['color'] = 0
white_df = pd.read_csv('winequality-white.csv', header = 0, sep= ';')
white_df['color'] = 1
red_white = pd.concat([red_df, white_df], axis=0, join='outer')
#split data
X_train, X_test, y_train, y_test = train_test_split(
    red_white.loc[:, 'fixed acidity':'quality'],
    red_white['color'],
    random_state =0
)

axes = pd.plotting.scatter_matrix(
    X_train,
    c=y_train,
    figsize=(12,12),
    marker='o',
    hist_kwds={'bins':20},
    s=10,
    alpha=0.8,
    cmap='copper_r',
)
for ax in axes.flatten():
    ax.xaxis.label.set_rotation(0)
    ax.xaxis.label.set_fontsize(8)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_fontsize(8)
    ax.yaxis.label.set_ha('right')

plt.gcf().subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.legend()
plt.show()


