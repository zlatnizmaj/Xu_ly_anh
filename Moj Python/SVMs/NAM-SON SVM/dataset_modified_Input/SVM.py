from csv import reader
from sklearn import svm
import numpy as np
import time


# Read datasets from CSV input file
def Read_file(file_name):
    dataset = list()
    with open(file_name, 'r', newline='', encoding='utf-8') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


start_time = time.time()

# testdataset
testdataset = Read_file('SVM_input.csv')
test_target= Read_file('SVM_target.csv')

n_epoch = 2
w = 0
r = 1
runs = [0]*n_epoch

# for j in range(len(runs)):
#     # # learning_rate = learning_rate + 0.05
#     # # n_epoch = n_epoch + 4
#     r = r + 5000
#     if (r + 5000) > 7536:
#         r = w
#         w += 300
#     traindataset = [testdataset[i] for i in range(r, r + 5000)]
#     train_target = [test_target[i] for i in range(r, r + 5000)]

from sklearn.model_selection import train_test_split

traindataset, X_test, train_target, y_test = train_test_split(testdataset, test_target, test_size=0.20)
clf = svm.SVC(kernel="rbf", decision_function_shape='ovo')
clf.fit(traindataset, train_target)
print(len(y_test))
print(len(X_test))
# test network
count = 0
total = 0
for i in range(0, len(X_test)):
    total += 1
    temp = clf.predict([X_test[i]])
    print('predicted:', temp, '== target:', y_test[i])
    if temp == y_test[i]:
        count += 1
accuracy = count * 100 / total
print('Accuracy: %s' % accuracy)
# runs[j] = accuracy

mean = sum(runs)/len(runs)
print("n_epoch: {}".format(n_epoch))
print("Mean_Accuracy: {}".format(mean))
print("Standard_Deviation: {}".format(np.std(runs, 0)))
time_taken = time.time() - start_time
print("total_time : {}".format(time_taken))

