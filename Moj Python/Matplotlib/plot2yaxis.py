"""
Sometimes, it is convenient to plot 2 data sets that have not the same range within the same plots.
One will use the left y-axes and the other will use the right y-axis.
With matplotlib, you need to create subplots and share the xaxes
"""
#  from pylab import figure, show, legend, ylabel
import matplotlib.pyplot as plt

# create the general figure
fig1 = plt.figure()

# and the first axes using subplot populated with data
ax1 = fig1.add_subplot(111)
line1 = ax1.plot([1, 3, 4, 5, 2], 'o-')
plt.ylabel("Left Y-Axis Data")

# now, the second axes that shares the x-axis with the ax1
ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)
line2 = ax2.plot([10, 40, 20, 30, 50], 'xr-')
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
plt.ylabel("Right Y-Axis Data")

# for the legend, remember that we used two different axes so, we need
# to build the legend manually
plt.legend((line1, line2), ("1", "2"))
plt.show()
