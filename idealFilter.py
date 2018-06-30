import matplotlib.pyplot as plt
import numpy
import math
import cv2

def ideallp(sx, sy, d0):
    hr = (sx)/2
    hc = (sy)/2

    x = numpy.arange(-hc, hc)
    y = numpy.arange(-hr, hr)

    [x, y] = numpy.meshgrid(x, y)
    mg = numpy.sqrt(x**2 + y**2)

    return numpy.double(mg<=d0)

I = cv2.imread('barca1.jpg')
ir, ic, channels= I.shape

print(ic,ir)

H = ideallp(ir, ic, 1000000000000)

G = numpy.fft.fftshift(numpy.fft.fft2(I[: ,: ,1]))



Ip = G * H
Im = numpy.abs(numpy.fft.fft2(numpy.fft.fftshift(Ip)))

plt.subplot(121)
plt.imshow(I)

plt.subplot(122)
plt.imshow(Im)
plt.imshow(numpy.rot90(Im,2))
rows,cols = Im.shape
#plt.imshow(cv2.getRotationMatrix2D((cols/2,rows/2),90,1))

plt.show()

