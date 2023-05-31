import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import cmath
import cv2

img = np.asarray(Image.open('aerial.tiff'))
print(repr(img))
f, imgplot = plt.subplots(3,5)

imgplot[0][0].imshow(img, cmap='gray')

#ex1.1.1
fimage = np.fft.fftshift(np.fft.fft2(img))
# print(repr(fimage))
imgplot[0][1].imshow(abs(fimage), cmap='gray')
imgplot[0][2].imshow(np.log(abs(fimage)), cmap='gray')

#ex1.1.3
# sigma = 2 # standard deviation of Gaussian
# ksize = len(fimage) # kernel size
# gaussian = cv2.getGaussianKernel(ksize, sigma) # create a normalized 1D kernel
# gaussian = gaussian.dot(gaussian.T) # create a 2D kernel by outer product

# # gaussian = np.fft.fftshift(gaussian)

# imgplot[1][0].imshow(gaussian, cmap='gray')

# lpfimage =  fimage * gaussian

# # print(repr([lpfimage[0]]))
# imgplot[1][1].imshow(abs(np.array(np.log(lpfimage))), cmap='gray')
# print(lpfimage)
# print(np.exp(lpfimage))
# print(fimage)

#Ideal low pass
radius = 40
mask = np.zeros_like(img)
cy = mask.shape[0] // 2
cx = mask.shape[1] // 2
cv2.circle(mask, (cx,cy), radius, (255,255,255), -1)[0]
lpfimage = np.multiply(fimage,mask) / 255
imgplot[1][0].imshow(mask, cmap='gray')

lpfshiftedmask = np.fft.fftshift(mask) # shift back the filtered image
lpmask = np.fft.fftshift(np.fft.ifft2(lpfshiftedmask)) # inverse Fourier transform
imgplot[1][1].imshow(np.log(lpmask.astype(np.uint8)+10), cmap='gray') # convert to uint8 and display
imgplot[1][2].plot(lpmask[len(lpmask)//2])

imgplot[1][3].imshow(abs(np.array(np.log(lpfimage + 1))), cmap='gray')


lpfshiftedimage = np.fft.ifftshift(lpfimage) # shift back the filtered image
lpimage = np.fft.ifft2(lpfshiftedimage) # inverse Fourier transform
imgplot[1][4].imshow(lpimage.clip(0,255).astype(np.uint8), cmap='gray') # convert to uint8 and display

#Ideal high pass
radius = 15
maskhp = np.full_like(img,255)
cy = maskhp.shape[0] // 2
cx = maskhp.shape[1] // 2
cv2.circle(maskhp, (cx,cy), radius, (0,0,0), -1)[0]
lpfimage = np.multiply(fimage,maskhp) / 255
imgplot[2][0].imshow(maskhp, cmap='gray')

hpfshiftedmask = np.fft.ifftshift(maskhp) # shift back the filtered image
hpmask = np.fft.fftshift(np.fft.ifft2(hpfshiftedmask)) # inverse Fourier transform
imgplot[2][1].imshow(np.log(hpmask.astype(np.uint8)+2).clip(0,255), cmap='gray') # convert to uint8 and display

imgplot[2][2].plot(hpmask[len(hpmask)//2][len(hpmask[0])//3:2*len(hpmask[0])//3])

imgplot[2][3].imshow(abs(np.log(lpfimage + 1)), cmap='gray')

lpfshiftedimage = np.fft.ifftshift(lpfimage) # shift back the filtered image
lpimage = np.fft.ifft2(lpfshiftedimage) # inverse Fourier transform
imgplot[2][4].imshow(lpimage.clip(0,255).astype(np.uint8), cmap='gray') # convert to uint8 and display

plt.show()

