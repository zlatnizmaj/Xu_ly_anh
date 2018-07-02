"""
numpy trang web machinelearningcoban.com, fundaml.com

"""
#  biến numpy là các biến mutable
import numpy as np
# khoi tao mang 1 chieu
x = np.array([1, 2, 3])
y = np.array([5, 6, 7], dtype= np.float)

print(x)
print(y)
#help(np.array)
print(type(x[0]))

# khoi tao vecto 0
zero_array = np.zeros(3)
print(zero_array)
# khoi tao vecto 1
one_array = np.ones(3)
print(one_array)

# hàm đặc biệt numpy.zeros_like và numpy.ones_like
# giúp tạo các mảng 0 và mảng 1 có số chiều giống như chiều của biến số.
x = np.array([1, 2, 3])
zero_array_like = np.zeros_like(x)
print(zero_array_like)

y = np.array([4, 5, 6])
print(y)
one_array_like = np.ones(x)
print(one_array_like)

# Array of zeros with the same shape and type as a.
# Return array of given shape and type as given array, with zeros,
# numpy.zeros_like method
array = np.arange(10).reshape(5, 2)
print("Original array : \n", array)

b = np.zeros_like(array, float)
print("\nMatrix b : \n", b)

array = np.arange(8)
c = np.zeros_like(array)
print("\nMatrix c: \n", c)

array = np.arange(4).reshape(2, 2)
c = np.zeros_like(array, dtype = 'float')
print("\nMatrix  : \n", c)


#  tạo mảng các số nguyên từ 0 đến n-1 (n số tổng cộng)
m = np.arange(0, 10, 2) #arrange(start, stop, step)
print('\n', m)
print(np.arange(3))
print(np.arange(2,5))
print(np.arange(5, 1, -0.9))

a = np.arange(0, 11)
b = 2
x = b**a #
print(x)

c = [3]*10
# for i in range(1,11):
#     c.append(3)
c[9] = 1.5
y = np.array(c)
print(y)