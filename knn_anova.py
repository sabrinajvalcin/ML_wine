import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.ticker import FormatStrFormatter

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
#select 25% of features based on ANOVA statistic, choosing features with highest variance
# f_classif calculates ANOVA F values between label/feature
#lets choose the three features with the highest variance
select = SelectPercentile(percentile=25)

select.fit(X_train, y_train)

#transform_training set
X_train_selected = select.transform(X_train)

#find out which features have been selected
mask = select.get_support()
print(mask)
#visualize the mask --black is true and white is false
features = red_white.loc[:, 'fixed acidity':'quality'].columns[mask]


plt.matshow(mask.reshape(1,-1), cmap='gray_r')
plt.xlabel('Feature Index')
plt.yticks(ticks=[])
plt.xticks(np.arange(0,12, step=1))
plt.show()

plt.clf()


#plot scatter matrix for selected features
axes = pd.plotting.scatter_matrix(
    X_train[features],
    c=y_train,
    figsize=(15,15),
    marker='o',
    hist_kwds={'bins':20},
    s=10,
    alpha=0.8,
    cmap='copper_r',
)
for ax in axes.flatten():
    ax.yaxis.label.set_rotation(90)
    ax.yaxis.label.set_ha('right')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

plt.gcf().subplots_adjust(wspace=0, hspace=0)
#plt.tight_layout()
plt.show()

#Compare KNN of data with all features vs data with selected features
knn_all = KNeighborsClassifier(n_neighbors=1)
knn_all.fit(X_train, y_train)
y_predict_all_test = knn_all.predict(X_test)
all_features_test_accuracy = np.mean(y_predict_all_test == y_test)

y_predict_all_train = knn_all.predict(X_train)
all_features_train_accuracy = np.mean(y_predict_all_train == y_train)
accuracy = []
num_neighbors = list(range(1, 11))

for i in num_neighbors:
    knn_select = KNeighborsClassifier(n_neighbors=i)
    knn_select.fit(X_train[features], y_train)
    y_predict_select_test = knn_select.predict(X_test[features])
    selected_features_accuracy_test = np.mean(y_predict_select_test == y_test)
    accuracy.append(selected_features_accuracy_test)
    
plt.plot(num_neighbors, accuracy)
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
