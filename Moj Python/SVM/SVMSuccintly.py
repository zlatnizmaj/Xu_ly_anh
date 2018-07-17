import numpy as np

x = [3, 4]
normOfx = np.linalg.norm(x)
print(normOfx) # 5.0 do dai vecto x, length,

# compute the direction of vector x

def direction(x):
    return x/np.linalg.norm(x)

w = direction(x)
print(w)
