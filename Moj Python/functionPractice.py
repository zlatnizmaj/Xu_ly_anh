"""
Docstring function:
The first statement in the body of a function is usually a string
which can be accessed with function_name.__doc__
This statement is called Docstring.
"""
import numpy as np


def hello(name='everybody'):
    """ Greets a person """
    print('Hello ' + name + '!')


print("The docstring of the function 'hello()': " + hello.__doc__)
print(__doc__)


def return_list():
    list01 = np.array(np.arange(0, 10, 1))
    list02 = np.array(np.arange(20, 30, 2))
    single_value = 100
    return list01, list02, single_value


def get_training_examples():
    X1 = np.array([[8, 7], [4, 10], [9, 7], [7, 10],
                   [9, 6], [4, 8], [10, 10]])
    y1 = np.ones(len(X1))
    X2 = np.array([[2, 7], [8, 3], [7, 5], [4, 4],
                   [4, 6], [1, 3], [2, 5]])
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2


l1, l2, s = return_list()
print(l1)
print('\n', s)

X1, y1, X2, y2 = get_training_examples()
#  nump.vstack() Stack arrays in sequence vertically (row wise)
#  Rebuilds arrays divided by vsplit()
print(X1, '\n\n', X2)
print('vstack X1 and X2: \n')
X_vstack = np.vstack((X1, X2))
X_hstack = np.hstack((X1, X2))
print(X_vstack)
print(X_hstack)
