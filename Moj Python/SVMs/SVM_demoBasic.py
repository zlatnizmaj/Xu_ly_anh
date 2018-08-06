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
# print(yy)

X_plot = np.c_[xx.ravel(), yy.ravel()]
# print(xx.ravel())
# print(yy.ravel())  # chuyen matrix thanh vector hang
# print('\n\n', X_plot)

