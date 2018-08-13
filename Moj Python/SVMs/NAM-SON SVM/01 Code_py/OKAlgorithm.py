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
import itertools
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
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
# scatter_matrix(dataset_input)

# Split-out validation dataset
svm_input = dataset_input.values
svm_target = dataset_target.values
# X = array[:, 0:4]
# Y = array[:, 4]
validation_size = 0.25
seed = 7
X_train, X_validation, y_train, y_validation = model_selection.train_test_split(svm_input, svm_target.ravel(), test_size=validation_size, random_state=seed)

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
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

model = SVC()
SVM = SVC(kernel="rbf", decision_function_shape='ovo')
SVM.fit(X_train, y_train)
predictions = SVM.predict(X_validation)
# print(X_validation, y_validation)
print('accuracy:\n', accuracy_score(y_validation, predictions))
print('confusion_matrix:\n', confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))

# print(X_validation,Y_validation), lưu ra model, mỗi lần change
# from pickle import dump
# filename = 'finalized_model.sav'
# dump(SVM, open(filename, 'wb'))

# confusion matrix
# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1, keepdims = True)
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="red" if cm[i, j] > thresh else "black")
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
#
# # Plot non-normalized confusion matrix
# cnf_matrix = confusion_matrix(y_validation, predictions)
# class_names = [0, 1]
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='Confusion matrix, without normalization')
#
# # Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')
#
# plt.show()

# Set the parameters by cross-validation
# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
#                      'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}]

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}]
scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_validation, clf.predict(X_validation)
    print(classification_report(y_true, y_pred))
    print()

time_taken = time.time() - start_time
print("total_time : {}".format(time_taken))









