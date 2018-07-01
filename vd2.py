import numpy as np
import cv2
import matplotlib.pyplot as plt
im=cv2.imread('D:\ANH.jpg')

plt.subplot(221)
plt.imshow(im[:,:,0])
#cv2.imshow('Red',im[:,:,2])

plt.subplot(222)
im[:,:,0]=np.log(im[:,:,0]+1)
plt.imshow(im)

plt.subplot(223)
plt.imshow(im[:,:,1])

plt.subplot(224)
#plt.imshow(im[:,:,1])=np.
plt.imshow(imgGray,cmap='gray')
#plt.imshow(im)
plt.show()
cv2.waitKey(0)