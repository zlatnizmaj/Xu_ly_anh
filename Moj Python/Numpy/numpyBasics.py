import numpy as np
print(np.__version__)

a = np.array([1, 2])
print(type(a[0]))

b = np.array([1, 2, 3], dtype=np.float)
print(type(b[2]))

print(2**10)

x = 2 ** np.arange(11)
print(x[10])
y = 3 * np.ones(10)
y[9] = 3/2

print(x, '\n', y)

# mảng toàn giá trị 0 hoặc 1
zeros = np.zeros(5)
print(zeros)

ones = np.ones(5)
print(ones)

print(np.zeros_like(x))
print(np.ones_like(x))

# Cap so cong
arith_sequence = np.arange(3, 6)
print(arith_sequence)
print(np.arange(0, 1, 0.1))
print(np.arange(5, 1, -0.9))

# Truy cap mang 1 chieu
print(x.shape)
d = x.shape[0]
print("Size of array x: ", d)

# -d <= i <= d-1
print(x[-11], x[-1])

'''
Thay toàn bộ các phần tử của mảng bằng trung bình cộng các phần tử trong mảng đó, 
sử dụng vòng for. 
Hàm này không trả về biến nào mà chỉ thay đổi 
các giá trị của biến đầu vào x.
'''

