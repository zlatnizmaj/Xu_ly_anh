"""
follow link:
https://sadanand-singh.github.io/posts/svmpython/
"""

import pandas as pd
import numpy as np
from sklearn import svm, datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two feature: sepal length, sepal width
y = iris.target

'''
test_list = [[1, 2, 3], [4, 5, 6]]
test_array = np.array(test_list)
print('test_list is: ', test_list)
print(test_list[0])
print('test_array is: \n', test_array)
print(test_array[:, 0])  # colume 1
print(test_array[0, :])  # row 1
'''
# print(X)
# print(y)
# print(X.shape)  # (150, 2)
# print(y.shape)  # (150,)
# print(X[:, 0].min()) # 4.3

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# print(x_min, x_max)
# print(y_min, y_max)

h = (x_max / x_min)/100
# print(h)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# print(xx.shape)
# print(yy.shape)
# print(xx)
# print(yy)

X_plot = np.c_[xx.ravel(), yy.ravel()]
# print(xx.ravel())
# print(yy.ravel())  # chuyen matrix thanh vector hang
# print('\n\n', X_plot)

# Create the SVC model object
C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C, decision_function_shape='ovr').fit(X, y)
Z = svc.predict(X_plot)
Z = Z.reshape(xx.shape)
# print(xx.shape)
# print(Z)

# plot 'linear' SVM
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
# print(X,y)
plt.xlabel('Sepal lenght')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')

# Create the SVC model object
C = 1  # SVM regularization parameter
svc = svm.SVC(kernel='rbf', gamma=10, C=C, decision_function_shape='ovr').fit(X, y)

Z = svc.predict(X_plot)
Z = Z.reshape(xx.shape)

plt.subplot(122)
plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with RBF kernel')

#plt.show()

# using 5-fold cross validation to perform grid search
# to calculate optimal hyper-parameters
from sklearn.model_selection import train_test_split
from sklearn.model_selection import  GridSearchCV
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

# shuffle the dataset
X, y = shuffle(X, y, random_state=0)

print(X, '\n', y.shape)
# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# print(X_train.shape, '\n\n', X_test.shape)
# print(y_train.shape, '\n\n', y_test)
# G = np.array(range(10))
# P = np.array(range(10))
# print(G)
# print(G.reshape(1, 10))
# print(3*G)

# Set the parameter by cross-validation
parameters = [{'kernel': ['rbf'],
               'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
               'C': [1, 10, 100, 1000]},
              {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

print('# Tuning hyper-parameters')
print()

clf = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=5)
clf.fit(X_train, y_train)
