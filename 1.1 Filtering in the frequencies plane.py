import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math

img = np.asarray(Image.open('aerial.tiff'))
# print(repr(img))
f, imgplot = plt.subplots(2,3)

imgplot[0][0].imshow(img, cmap='gray')

#ex1.1.1
fimage = abs(np.fft.fftshift(np.fft.fft2(img)))
# print(repr(fimage))
imgplot[0][1].imshow(fimage, cmap='gray')
imgplot[0][2].imshow(np.log(fimage), cmap='gray')

#ex1.1.3
gaussian = [[0]*len(fimage[0]) for i in range(len(fimage))]
σ = 0.03
k = 1/2*math.pi*σ**2

for i in range(len(gaussian)):
    for j in range(len(gaussian[0])):
        gaussian[i][j] = k*np.exp(-((i-int(len(gaussian)/2))**2  + (j-int(len(gaussian[0])/2))**2)/2*σ**2)

# gaussian = np.fft.fftshift(gaussian)
imgplot[1][0].imshow(gaussian, cmap='gray')

logfimage = np.log(fimage)
norm = np.linalg.norm(gaussian) 
ngaussian = gaussian/norm
lpfimage = [[0]*len(fimage[0]) for i in range(len(fimage))]
for i in range(len(lpfimage)):
    for j in range(len(lpfimage[0])):
        lpfimage[i][j] =  logfimage[i][j] * ngaussian[i][j]

imgplot[1][1].imshow(lpfimage, cmap='gray')

# lpimage = np.fft.ifft2(lpfimage)
# imgplot[1][2].imshow(lpfimage, cmap='gray')

plt.show()

