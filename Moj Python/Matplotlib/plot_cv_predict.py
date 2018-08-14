"""
====================================
Plotting Cross-Validated Predictions
====================================

This example shows how to use `cross_val_predict` to visualize prediction
errors.

"""
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt

lr = linear_model.LinearRegression()
boston = datasets.load_boston()
y = boston.target

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
# predicted = cross_val_predict(lr, boston.data, y, cv=10)
#
# fig, ax = plt.subplots()
# ax.scatter(y, predicted, edgecolors=(0, 0, 0))
# ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# plt.show()
from mpl_toolkits.mplot3d import Axes3D
from generate_dataset import generate_dataset as gd
from matplotlib import pyplot as plt

predictor_coeffs =[1, 1]
std_dev = 10
n = 100

X, Y = gd(predictor_coeffs, n, std_dev)


f, [[p1, p2], [p3, p4]] = plt.subplots(2,2)

p1.scatter(X[:,0], Y)
p1.set_title("x1")

p2.scatter(X[:,1], Y)
p2.set_title("x2")

p3 = f.add_subplot(223, projection='3d')
p3.scatter(X[:,0], X[:,1], Y)
p3.set_title("x1, x2, y")

plt.show()
