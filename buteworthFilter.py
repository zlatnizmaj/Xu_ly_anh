import matplotlib.pyplot as plt
import numpy
import math
import cv2

def buteworthlp(sx, sy, d0, n):
    hr = (sx)/2
    hc = (sy)/2

    x = numpy.arange(-hc, hc)
    y = numpy.arange(-hr, hr)

    [x, y] = numpy.meshgrid(x, y)
    mg = numpy.sqrt(x**2 + y**2)

    return 1/(1+(mg/d0)**(2*n)) # ko can 1./, python tu nhan chia ma tran

I = cv2.imread('lena_color.tiff')
ir, ic, channels= I.shape

print(ic,ir)

H = buteworthlp(ir, ic, -10, 2)

G = numpy.fft.fftshift(numpy.fft.fft2(I[: ,: ,1]))



Ip = G * H
Im = numpy.abs(numpy.fft.fft2(numpy.fft.fftshift(Ip)))

plt.subplot(121)
plt.imshow(I)

plt.subplot(122)
plt.imshow(Im)
plt.imshow(numpy.rot90(Im,2))
rows,cols = Im.shape


plt.show()

