from succinctly.datasets import get_dataset, linearly_separable as ls
import numpy as np


def hypotesis(x, w):
    return np.sign(np.dot(w, x))

#  Make prediction on all data points
#  and return the ones that are misclassified


def predict(hypothesis_function, X, y, w):
    predictions = np.apply_along_axis(hypothesis_function, 1, X, w)
    misclassified = X[y != predictions]
    return misclassified


#  Pick one misclassified example randomly
#  and return it with its expected label.


def pick_one_from(misclassified_examples, X, y):
    np.random.shuffle(misclassified_examples)
    x = misclassified_examples[0]
    index = np.where(np.all(X == x, axis=1))
    return x, y[index]


def perceptron_learning_algorithm(X, y):
    w = np.random.rand(3)  # can also be initialized at zero.
    misclassified_examples = predict(hypotesis, X, y, w)
    while misclassified_examples.any():
        x, expected_y = pick_one_from(misclassified_examples, X, y)
        w = w + x * expected_y  # update rule
        misclassified_examples = predict(hypotesis, X, y, w)
    return w


np.random.seed(88)
# print(np.random.rand(4))

X, y = get_dataset(ls.get_training_examples)

# print(len(X))
# print(X, y)
print(X)
print(X.shape)

# transform X into an array of augmeted vectors.
X_augmented = np.c_[np.ones(X.shape[0]), X]
w = perceptron_learning_algorithm(X_augmented, y)

print(w)
# print(X_augmented)
# print(np.r_[-1:1:5j])
# print(np.linspace(-1, 1, num=5, endpoint=False, retstep=True))
