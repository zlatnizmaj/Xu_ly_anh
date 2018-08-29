import numpy as np
import cv2
import matplotlib.pyplot as plt

im = cv2.imread('hello.png')

plt.subplot(221)
plt.imshow(im)

plt.subplot(222)

im[:,:,0] = np.log(im[:,:,0]+1)
plt.imshow(im[:,:,0], cmap='gray')

plt.subplot(223)
plt.imshow(im[:,:,1])

plt.subplot(224)
plt.imshow(im[:,:,2])

plt.imshow(im, cmap='gray')
plt._show()

cv2.waitKey(0)
cv2.destroyAllWindows()

# comment 