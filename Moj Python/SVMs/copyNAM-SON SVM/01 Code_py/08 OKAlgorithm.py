# -*- coding: utf-8 -*-
"""
Đại học Sư phạm Thành phố Hồ Chí Minh

BÀI TẬP CUỐI KỲ
Chuyên ngành: Khoa học máy tính
Khóa: ĐHSP-K28
Môn học: Giải thuật nâng cao
Giáo viên hướng dẫn: TS. Bùi Thanh Hùng
Nhóm học viên:
    1. Nguyễn Phương Nam
    2. Nguyễn Quý Sơn

"""
print(__doc__)
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
from pickle import dump
import pickle as pkl
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC  # build lại giải thuật

path_files_CSV = "../dataset_modified_Input/"
start_time = time.time()
# Load dataset
dataset = pandas.read_csv(path_files_CSV + 'Input_dataset.csv')

# shape
print(dataset.shape)

# head
print(dataset.head())

# descriptions
print(dataset.describe())

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2, 4), sharex=False, sharey=False)
plt.show()

# histograms
dataset.hist()
plt.show()

# scatter plot matrix
# scatter_matrix(dataset)

# Split-out validation dataset
array = dataset.values
X_input = array[:, 0:6]
y_target = array[:, 6]
print(X_input[0])
validation_size = 0.20
seed = 7
X_train, X_validation, y_train, y_validation = model_selection.train_test_split(X_input, y_target,
                                                                                test_size=validation_size,
                                                                                random_state=seed)

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
    print(kfold)
    # for train, test in kfold.split(X_train):
    #     print('train: %s, test: %s' % (train, test))
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    print('All results (k-fold):\n', results)

model = SVC()
SVM = SVC(kernel="rbf", decision_function_shape='ovo')
SVM.fit(X_train, y_train)

predictions = SVM.predict(X_validation)
print('Accuracy:\n', accuracy_score(y_validation, predictions))
print('Confusion_matrix:\n', confusion_matrix(y_validation, predictions))
print("Detailed classification report:\n")
print(classification_report(y_validation, predictions))


# plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="red" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Plot non-normalized confusion matrix
cnf_matrix = confusion_matrix(y_validation, predictions)
class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

# tuned_parameters = [{'C': [0.01, 0.1, 1, 10, 100],
#                      'gamma': [0.5, 1, 2, 3, 4]}]
# clf = GridSearchCV(SVC(kernel='rbf'), tuned_parameters, cv=10, scoring='accuracy', return_train_score=True)
# clf.fit(X_train, y_train)
#
# print(clf.cv_results_)
# print(clf.best_params_)
# print(confusion_matrix(y_validation, clf.best_estimator_.predict(X_validation)))
# print(clf.best_estimator_.score(X_validation, y_validation))


# print(X_validation,Y_validation), lưu ra model, mỗi lần change

# filename = path_files_CSV + 'SVM_finalized_model.sav'
#
# SVM_model_pkl = open(filename, 'wb')
# dump(SVM, SVM_model_pkl)
# SVM_model_pkl.close()

# Loading the saved decision tree model pickle
# SVM_model_pkl = open(filename, 'rb')
# SVM_model = pkl.load(SVM_model_pkl)
# print("Loaded SVM model :: ", SVM_model)

# time for run training
time_taken = time.time() - start_time
print("total_time : {}".format(time_taken))









