from succinctly.datasets import get_dataset, linearly_separable as ls
import numpy as np

np.random.seed(88)
#  print(np.random.rand(4))

X, y = get_dataset(ls.get_training_examples)

# print(X, y)
#  transform X into an array of augmeted vectors.
X_augmented = np.c_[np.ones(X.shape[0]), X]
# print(X_augmented)
print(np.r_[-1:1:5j])
print(np.linspace(-1, 1, num=5, endpoint=False, retstep=True))
