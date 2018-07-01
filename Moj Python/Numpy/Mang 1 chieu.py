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

#  tạo mảng các số nguyên từ 0 đến n-1 (n số tổng cộng)
m = np.arange(0, 10, 2) #arrange(start, stop, step)
print(m)
print(np.arange(3))
print(np.arange(2,5))
print(np.arange(5, 1, -0.9))