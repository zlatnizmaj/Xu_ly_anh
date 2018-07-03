# graphing in python with Matplotlib

# Ploting a line

#import the required module
import matplotlib.pyplot as plt
import numpy as np

# # x axis values
# x = [1, 2, 3]
# # corresponding y axis values
# y = [2, 4, 1]
#
# # plotting the points
# plt.plot(x, y)
#
# # naming the x axis
# plt.xlabel('x - axis')
# plt. ylabel('y - axis')
#
# # giving the tite to my graph
# plt.title('My first graph!')
#
# # function to show the plot
# plt.show()
#
# ############################
# # Plotting two or more lines on same plot
# # line 1 points
# x1 = [1, 2, 3]
# y1 = [2, 4, 1]
# # plotting the line 1 points
# plt.plot(x1, y1, label = 'line 1')
#
# # line 2 points
# x2 = [1, 2, 3]
# y2 = [4, 1, 3]
# # plotting the line 2 points
# plt.plot(x2, y2, label = 'line 2')
#
# # naming the x axis
# plt.xlabel('x - axis')
# plt. ylabel('y - axis')
#
# # giving the tite to my graph
# plt.title('Two lines on same graph!')
#
# # show a legend on the plot
# plt.legend()
#
# # function to show the plot
# plt.show()
# """
# Here, we plot two lines on same graph.
# We differentiate between them by giving them a name(label) which is passed as an argument of .plot() function.
# The small rectangular box giving information about type of line and its color is called legend.
# We can add a legend to our plot using .legend() function.
# """
# #############################
# # Customization of Plots
# x = np.arange(1, 7)
# y = [2, 4, 1, 5, 2, 6]
# plt.plot(x, y, color = 'green', linestyle = 'dashed', linewidth = 3, marker = '*', markerfacecolor = 'red', markersize = 12)
#
# # setting x and y axis range
# plt.xlim(1, 8)
# plt.ylim(1, 8)
#
# # naming the x axis
# plt.xlabel('x - axis')
# plt. ylabel('y - axis')
#
# # giving the tite to my graph
# plt.title('Some cool customizations!')
#
# # show a legend on the plot
# # plt.legend() # No handles with labels found to put in legend
#
# # function to show the plot
# plt.show()
#
# ##############################
# # Bar Chart
# # x-coordinates of left sides of bars
# left = np.arange(1, 6)
#
# # height of bars
# height = [10, 24, 36, 40, 5]
#
# # labels for bars
# tick_label = ['one', 'two', 'three', 'four', 'five']
#
# # plotting a bar chart
# plt.bar(left, height, tick_label = tick_label, width = .8, color = ['red', 'green'])
#
# # naming the x axis
# plt.xlabel('x - axis')
# plt. ylabel('y - axis')
#
# # giving the tite to my graph
# plt.title('My bar chart!')
# plt.show()
# # plt.bar() function to plot a bar chart
# ##########################################
# # Histogram
# # plt.hist() function to plot a histogram
#
# # frequencies
# ages = [2,5,70,40,30,45,50,45,43,40,44,
#         60,7,13,57,18,90,77,32,21,20,40]
# # setting the ranges and no.of intervals
# range = (0, 100)
# bins = 10
# # plotting a histogram
# plt.hist(ages, bins, range, color = 'green', histtype= 'bar', rwidth = .8)
#
# plt.xlabel('age')
# plt.ylabel('No. of people')
# plt.title('My histogram')
# plt.show()
# """
# frequencies are passed as the ages list.
# Range could be set by defining a tuple containing min and max value.
# Next step is to “bin” the range of values—that is,
# divide the entire range of values into a series of intervals—and
# then count how many values fall into each interval.
# Here we have defined bins = 10. So, there are a total of 100/10 = 10 intervals.
# """
# ##########################################
# # Plotting Scatter Plot (do thi phan tan)
# # x-axisvalues
# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# # y-axis values
# y = [2, 4, 5, 7, 6, 8, 9, 11, 12, 12]
#
# # plotting points as a scatter plot
# plt.scatter(x, y, label="stars", color="green",
#             marker="*", s=30)
#
# # x-axis label
# plt.xlabel('x - axis')
# # frequency label
# plt.ylabel('y - axis')
# # plot title
# plt.title('My scatter plot!')
# # showing legend
# plt.legend()
#
# # function to show the plot
# plt.show()
###########################################
# Pie-chart
# defining labels
activities = ['eat', 'sleep', 'work', 'play']

# portion covered by each label
slices = [3, 7, 8, 6]
colors = ['y', 'r', 'g', 'b']

# plotting the pie chart
plt.pie(slices, labels = activities, colors = colors, startangle = -45,
        shadow = True, explode=(0,0,0.1,0), radius = 1.2, autopct= '%1.2f%%')

plt.legend()
plt.show()

############################################
# Plotting curves of given equation y = sin(x)
x = np.arange(0, 2 *(np.pi), 0.1)
y = np.sin(x)

plt.plot(x,y)
plt.show()