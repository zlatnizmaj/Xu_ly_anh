"""
In python, matrix is a nested list (list in list form table)
A list is created by placing all the items (elements)
inside a square bracket [ ], separated by commas
"""
import numpy as np
from numpy import *

# Create a matrix in python
A = [[1.5, -2], [.5, .1]]
print(A)
print(A[1][1])

# Create a dynamic matrix using for loop in python
# matrix n*m
# first create a list of n elements (say, of n zeros)
# then make each of the elements a link to anothe one-dimensional list of m ele
print("=================")
print("Matrix using loop")
n = 3
m = 4
LoopMatrix = [1] * n
print(LoopMatrix)

for i in range(n):
    LoopMatrix[i] = [0] * m
print(LoopMatrix)
print("=================")

# matrix using numpy
print("Matrix using numpy")
x = range(16)
x = reshape(x, (4, 4))
print(x)
print("=================")

# access elements of matrix
# List index
print(x[0])
print(x[0][3])
print(x[-1]) # a b c (index -3 -2 -1)
x[0][3] = 23
print(x)
print("++++++++++++++")


# thu vien numpy
# slice matrix, (start:stop:step)
print (np.matrix([[1, 1], [2, 2]]))
B = np.matrix('1 2 3; 4 5 6')
print(B)
print(B[:2, :2], '\n', B[1,2])

b_row = B[0] # b_row vecto dong
print (b_row)
#bprint[0, 1] = 'MrGao'
print(b_row[0,0]) # in ra element thu 2 cua vecto dong b_row (0,1)
print("************")

a = [['Roy',80,75,85,90,95],
     ['John',75,80,75,85,100],
     ['Dave',80,80,80,90,95]]

b=a[0]
print(b)

b[1]=75
print(b)

a[2]=['Sam',82,79,88,97,99]
print(a)

a[0][4]=95
print(a)

# add elements in matrix
print('============')
print("Add element in matrix")
# add one row to a matrix using append() method
# and add a item using insert() method by importing numpy library.
print(B)
B = array(B)
B = append(B,[7, 8, 9], 0)
print(B)

