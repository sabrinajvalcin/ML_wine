import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn.tree import export_graphviz

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

#track accuracy with tree depth. when does overfitting occcur?
tree_accuracy=[]
feature_names = X_train.columns
feature_importance = []
for depth in range(1,30):
    tree = DecisionTreeClassifier(max_depth=depth, random_state=0)
    tree.fit(X_train, y_train)
    export_graphviz(tree, out_file=f'tree.dot', class_names=["red", "white"], max_depth = 4, feature_names=feature_names, impurity=False, filled=True)
    feature_importance.append(tree.feature_importances_)
    plt.show()
    test_accuracy = tree.score(X_test, y_test)
    train_accuracy = tree.score(X_train, y_train)
    tree_accuracy.append({'depth':depth, 'training_accuracy': train_accuracy, 'test_accuracy' : test_accuracy })

tree_accuracy_df = pd.DataFrame(tree_accuracy)

plt.plot(
    tree_accuracy_df['depth'],
    tree_accuracy_df['training_accuracy'],
    label = 'Training Accuracy',
    c= 'maroon',
    marker = 'o'
)
plt.plot(
    tree_accuracy_df['depth'],
    tree_accuracy_df['test_accuracy'],
    label = 'Test Accuracy',
    c= 'orange',
    marker = '^'
)
plt.xlabel('Tree Depth')
plt.ylabel('Model Accuracy')
plt.legend()
plt.show()

#feature importance chart when max_depth = 7
plt.barh(feature_names, feature_importance[6])
plt.title('Feature Importance when max depth is 7')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

