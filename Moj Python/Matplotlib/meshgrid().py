import matplotlib.pyplot as plt
import numpy as np

xvalues = np.array([0, 1, 2, 3, 4])
yvalues = np.arange(0, 5)

xx, yy = np.meshgrid(xvalues, yvalues)

plt.plot(xx, yy, marker='.', color='k', linestyle='none')
plt.show()