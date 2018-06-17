import cv2 # thu vien xu ly anh
import numpy as np
import random
from scipy import ndimage
import matplotlib.pyplot as plt

I = cv2.imread('hello.png')

II = I[:,:,0]
id = np.array([[0,0,0],[0,1,0],[0,0,0]])

f = np.ones([3,3])/9 # nhan chap smoothing

hb = 3*id - 2*f # nhan chap
hbI = ndimage.convolve(II, f)

# plt.subplot(121)
# plt.imshow(I)
#
# plt.subplot(122)
plt.imshow(hbI, cmap= 'gray')
plt.show()

####################
##2D Convolutions in Python
#image = cv2.imread('clock.jpg', cv2.IMREAD_GRAYSCALE.astype(float)) / 255.0

#kernel = np.array([[1,0,-1],
                   #[1,0,-1],
                   #[1,0,-1]])
#filtered = cv2.filter2D(src=image, kernel=kernel)

#cv2.waitKey(0)