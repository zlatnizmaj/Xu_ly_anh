# basic SVM

import numpy as np
import math
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

x = [3, 4]
normOfx = np.linalg.norm(x) # length, norm, magnitude of vector x(x1, x2)
print(normOfx)

# Compute the direction of a vector x.


def direction(x):
    return x/np.linalg.norm(x)


u = np.array([3, 4])  # toa do vector u
w = direction(u)  # vector huong cua u, w(3/5, 4/5)

print(w)

# same direction will have same direction vector
u_1 = np.array([3, 4])
u_2 = np.array([30, 40])

print(direction(u_1))
print(direction(u_2))

# the norm of a direction vector is always 1
print(np.linalg.norm(w))

# dot product is also call scalar product, tich vo huong 2 vector
# Geometric definition


def geometric_dot_product(x, y, theta):  # theta la goc giua 2 vector
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    return x_norm * y_norm * math.cos(math.radians(theta))


# we must now theta
theta = 45
x = [3, 5]
y = [8, 2]

print(geometric_dot_product(x, y, theta))

# Algebraic def of dot product


def dot_product(x, y):
    result = 0
    for i in range (len(x)):
        result += x[i] * y[i]
    return result


print(dot_product(x, y))

fig = plt.figure()
ax = plt.axes()

x = np.linspace(0, 10, 100)
ax.plot(x, -0.4*x - 9)

plt.show()
