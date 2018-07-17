import numpy as np


def update_rule(expected_y, w, x):
    w = w + x * expected_y
    return w


def hypothesis(x, w):
    return np.sign(np.dot(w, x))


x = np.array([1, 2, 7])  # w0 = 1
expected_y = -1
w = np.array([4, 5, 3])  # b = 4

print(hypothesis(w, x))  # The predicted y is 1

w = update_rule(expected_y, w, x)  # We apply the update rule.

print(hypothesis(w, x))  # The predicted y is -1
