import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
import time

# path_files_CSV = "../dataset_modified_Input/"

start_time = time.time()

# read input dataset
input_data = pd.read_csv('SVM_Input.csv', header=None, index_col=None)
input_target = pd.read_csv('SVM_Target.csv', header=None, index_col=None)

# print(input_data)
# df.drop(df.index[0], inplace=True)
svm_input = input_data.values.tolist()
print(svm_input[0])
#
svm_targer = input_target.values.tolist()
print(len(svm_targer))
print(svm_targer)

# print(input_data.head())
n_epoch = 2
w = 0
r = 0
runs = [0]*n_epoch

# for j in range(len(runs)):
#     # # learning_rate = learning_rate + 0.05
#     # # n_epoch = n_epoch + 4
#     r = r + 1000
#     if (r + 1000) > 7535:
#         r = w
#         w += 30
# X_train = input_data[0:5000]
# Y_train = input_target[0:5000]
#


X_train, X_test, y_train, y_test = train_test_split(svm_input, svm_targer, test_size=0.20)

clf = svm.SVC(kernel="rbf", decision_function_shape='ovo')
clf.fit(X_train, y_train)

#
# # test network
count = 0
total = 0
for i in range(0, len(svm_input)):
    total += 1
    temp = clf.predict([svm_input[i]])
    print('predict: ', temp, '-- target:', svm_targer[i])
    if temp == svm_targer[i]:
        count += 1
accuracy = count * 100 / total
print('Accuracy: %s' % accuracy)
runs[j] = accuracy

mean = sum(runs)/len(runs)
print("n_epoch: {}".format(n_epoch))
print("Mean_Accuracy: {}".format(mean))
print("Standard_Deviation: {}".format(np.std(runs, 0)))

time_taken = time.time() - start_time
print("total_time : {}".format(time_taken))

