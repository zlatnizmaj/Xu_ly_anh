import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import time

path_files_CSV = "../dataset_modified_Input/"

start_time = time.time()

# read input dataset
input_data = pd.read_csv(path_files_CSV + 'SVM_input.csv')
input_target = pd.read_csv(path_files_CSV + 'SVM_target.csv')
print(input_data.head())

# prepare dataset for svm input
# svm_input = input_data.values.tolist()
svm_input = np.array(input_data)
print(svm_input.shape)

svm_target = np.array(input_target)
# svm_target = input_target.values.tolist()
print(svm_target.shape)

n_epoch = 1
w = 0
r = 0
runs = [0]*n_epoch

for j in range(len(runs)):
    X_train, X_test, y_train, y_test = train_test_split(svm_input, svm_target.ravel(), test_size=0.25)
    print(j, ':', 'X_train shape:', X_train.shape, ', y_train shape:', y_train.shape,
          '\n', '   X_test shape:', X_test.shape, 'y_test shape:', y_test.shape, '\n',
          X_train[0, :])

    clf = svm.SVC(kernel="rbf", decision_function_shape='ovo')
    clf.fit(X_train, y_train)

    # test network
    count = 0
    total = 0
    for i in range(0, len(svm_input)):
        total += 1
        temp = clf.predict([svm_input[i]])
        # print('predict: ', temp, '-- target:', svm_target[i])
        if temp == svm_target[i]:
            count += 1
    accuracy = count * 100 / total
    print('Accuracy: %s\n' % accuracy, 'Score: ', clf.score(X_train,y_train))
    print('Get parameter: ', clf.get_params())
    runs[j] = accuracy

for i in range(0, len(X_test)):
    print(clf.predict([X_test[i]]), '===', y_test[i])
print('Score:', clf.score(X_test, y_test))

mean = sum(runs)/len(runs)
print("\nn_epoch: {}".format(n_epoch))
print("Mean_Accuracy: {}".format(mean))
print("Standard_Deviation: {}".format(np.std(runs, 0)))

time_taken = time.time() - start_time
print("total_time : {}".format(time_taken))

