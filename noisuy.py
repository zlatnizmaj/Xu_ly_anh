import numpy as np
import random
import cv2
from scipy import ndimage
from matplotlib import pyplot as plt
i=cv2.imread('D:\ANH.jpg')
ii=i[:,:,0]
id=np.array([[0,0,0],[0,1,0],[0,0,0]])
f=np.ones([5,5])/25
#hb=3*id-2*f
hbi=ndimage.convolve(ii,f)
plt.imshow(hbi)

plt.show()