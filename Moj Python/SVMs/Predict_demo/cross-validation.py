from sklearn import datasets, svm
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
svc = svm.SVC(C=1, kernel='linear')
BIGGER = svc.fit(X_digits[:-100], y_digits[0:-100]).score(X_digits[-100:], y_digits[-100:])
print(BIGGER)
