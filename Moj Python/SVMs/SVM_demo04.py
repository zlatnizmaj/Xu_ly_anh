import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC  # "Support vector classifier"
from mpl_toolkits import mplot3d


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


fig = plt.figure()

from sklearn.datasets.samples_generator import make_circles
X, y = make_circles(100, factor=.1, noise=.1)

clf = SVC(kernel='linear').fit(X, y)

# fig.add_subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

# fig.add_subplot(1, 2, 2)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
# plot_svc_decision_function(clf, plot_support=False)  # plot su dung Linear

r = np.exp(-(X ** 2).sum(1))  # ham Kernel RBF


def plot_3D(elev=30, azim=30, X=X, y=y):
    ax = plt.axes(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='autumn')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')


fig = plt.figure()

# interact(plot_3D, elev=[-90, 90], azip=(-180, 180), X=fixed(X), y=fixed(y))

# ax = fig.add_subplot(1, 2, 1)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

# ax = fig.add_subplot(1, 2, 2, projection='3d')  # 2
# plot_3D()  # 2

clf = SVC(kernel='rbf', C=1E6)
print(clf.fit(X, y))

print(clf.predict(([3, 1], [4, 2])))

plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=300, lw=1, facecolors='none')

plt.show()
