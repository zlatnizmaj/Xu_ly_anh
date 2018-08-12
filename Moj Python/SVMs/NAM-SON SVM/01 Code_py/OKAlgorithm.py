# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 20:58:24 2017
@author:

1. Nguyễn Phương Nam
2. Nguyễn Quý Sơn

DHSP - Khoa học máy tính K28
"""

# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))

import time
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC  # build lại giải thuật

path_files_CSV = "../dataset_modified_Input/"
names = ['Close', 'Index_Momentum', 'Index_Volatility',
         'Sector_Momentum',
         'Stock_Momentum', 'Stock_Volatility']

start_time = time.time()
# Load dataset
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# dataset = pandas.read_csv(url, names=names)
# dataset = pandas.read_csv('Iris.txt', names=names)

dataset_input = pd.read_csv(path_files_CSV + 'SVM_input.csv', names=names)
dataset_target = pd.read_csv(path_files_CSV + 'SVM_target.csv', header=None)

# shape
print(dataset_input.shape)

# head
print(dataset_input.head())

# descriptions
print(dataset_input.describe())

# class distribution
# print(dataset_input.groupby('class').size())

# box and whisker plots
# dataset_input.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
# plt.show()


# histograms
dataset_input.hist()
plt.show()

# scatter plot matrix
scatter_matrix(dataset_input)

# Split-out validation dataset
svm_input = dataset_input.values
svm_target = dataset_target.values
# X = array[:, 0:4]
# Y = array[:, 4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Spot Check Algorithms
models = []
# models.append(('LR', LogisticRegression()))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Test options and evaluation metric

scoring = 'accuracy'
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train,cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

model = SVC()
SVM = SVC()
SVM.fit(X_train, Y_train)
predictions = SVM.predict(X_validation)
print(X_validation, Y_validation)
print('accuracy:\n', accuracy_score(Y_validation, predictions))
print('confusion_matrix:\n',confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# print(X_validation,Y_validation), lưu ra model, mỗi lần change
from pickle import dump
filename = 'finalized_model.sav'
dump(SVM, open(filename, 'wb'))











