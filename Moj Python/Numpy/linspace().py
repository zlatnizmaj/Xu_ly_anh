"""
numpy.linspace() in Python
About :
numpy.linspace(start, stop, num = 50, endpoint = True, retstep = False, dtype = None) :
Returns number spaces evenly w.r.t interval.
Similiar to arange but instead of step it uses sample number.
Parameters :

-> start  : [optional] start of interval range. By default start = 0
-> stop   : end of interval range
-> restep : If True, return (samples, step). By deflut restep = False
-> num    : [int, optional] No. of samples to generate
-> dtype  : type of output array
Return :

-> ndarray
-> step : [float, optional], if restep = True
"""
# Code 1: explain linspace function
# numpy.linspace method

import numpy as np
import pylab as p

# restep set to True
print("B\n", np.linspace(2.0, 3.0, num=5, endpoint=False, retstep=True), "\n")

# to evaluate sin() in long range
# x = np.linspace(0, 2, 10)
# print("x vector:\n", x)
# print("A, sin(x)\n", np.sin(x))

# Graphical represnetation of numpy.linsapce() using matplotlib module- pylab

# start = 0
# end = 2
# smaples to generate = 10
x1 = np.linspace(0, 10, 5, endpoint=True, retstep=True)
x2 = np.linspace(0, 10, 5, endpoint=False, retstep=True)
print(11/5.0)
y1 = np.ones(10)
print(x1)
print(x2)
# p.plot(x1, y1, '*')
# p.xlim((-0.2, 1.8))
# p.show()