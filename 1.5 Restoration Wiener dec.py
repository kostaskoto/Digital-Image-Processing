import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math
import cv2
from scipy.signal import wiener

def addGaussianNoise(img, NSR):
    # img is the input gray image
    # NSR is the noise to signal ratio in dB
    # return the noisy image
    # The function adds Gaussian noise to the image
    # The function uses the formula: NSR = 10*log10(variance of noise / variance of signal)      

    noisyImg = np.zeros_like(img)
    #calculate the variance of the signal
    signalVariance = np.var(img)
    #calculate the variance of the noise
    noiseVariance = signalVariance / (10**(NSR/10))
    #calculate the standard deviation of the noise
    noiseStd = np.sqrt(noiseVariance)
    #add the noise to the image
    noisyImg = img + np.random.normal(0, noiseStd, img.shape)

    return noisyImg

def inverseFilter(img, filter):
    # img is the input gray image
    # filter is the filter to be applied
    # return the filtered image
    # The function uses inverse filter to remove the noise from the image   
    
    #apply the filter
    filteredImg = np.zeros_like(img)
    for i in range(len(img)):
        for j in range(len(img[0])):
            filteredImg[i][j] = img[i][j] + (1 / filter[i][j])
    
    return filteredImg

img = np.asarray(Image.open('factory.jpg').convert('L'))

f, imgplot = plt.subplots(2,3)

imgplot[0][0].imshow(img, cmap='gray')

fimage = np.fft.fftshift(np.fft.fft2(img))

# imgplot[0][1].imshow(np.log(abs(fimage)), cmap='gray')

sigma = 20 # standard deviation of Gaussian
gaussian1 = cv2.getGaussianKernel(len(fimage), sigma) # create a normalized 1D kernel
gaussian2 = cv2.getGaussianKernel(len(fimage[0]), sigma) # create a normalized 1D kernel
gaussian = gaussian1.dot(gaussian2.T) # create a 2D kernel by outer product\
gaussian = gaussian / gaussian.max() # normalize the kernel

imgplot[0][1].imshow(np.log(abs(gaussian)+1), cmap='gray')

# lpfshiftedmask = np.fft.ifftshift(gaussian) # shift back the filtered image
# lpmask = np.fft.ifft2(lpfshiftedmask)
# imgplot[0][2].imshow(np.log(lpmask.astype(np.uint8)+1), cmap='gray')


lpfimage =  fimage * gaussian
# imgplot[1][1].imshow(np.log(abs(lpfimage)+1), cmap='gray')
lpfshiftedimage = np.fft.ifftshift(lpfimage) # shift back the filtered image
lpimage = np.fft.ifft2(lpfshiftedimage)
imgplot[0][2].imshow(lpimage.astype(np.uint8), cmap='gray')

lpimagenoisy = addGaussianNoise(lpimage, 10)
imgplot[1][0].imshow(lpimagenoisy.astype(np.uint8), cmap='gray')

imageWiener = wiener(lpimagenoisy, 10, 10)
imgplot[1][1].imshow(imageWiener.astype(np.uint8), cmap='gray')

# imageInvFilterFourier = inverseFilter(np.fft.fftshift(np.fft.fft2(imageWiener)), gaussian)
# imgplot[1][2].imshow(imageInvFilterFourier.astype(np.uint8), cmap='gray')

# imageInvFilter = np.fft.ifft2(np.fft.ifftshift(imageInvFilterFourier))
# imgplot[1][3].imshow(imageInvFilter.astype(np.uint8), cmap='gray')


imageWiener = wiener(lpimagenoisy)
imgplot[1][2].imshow(imageWiener.astype(np.uint8), cmap='gray')

plt.show()