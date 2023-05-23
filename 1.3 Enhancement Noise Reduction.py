import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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

def addSaltAndPepperNoise(img, noiseDensity):
    # img is the input gray image
    # noiseDensity is the percentage of the pixels to be contaminated
    # return the noisy image
    # The function adds salt and pepper noise to the image
    # The function uses the formula: noiseDensity = (number of noisy pixels) / (total number of pixels) * 100%   
    
    noisyImg = np.zeros_like(img)
    #calculate the number of noisy pixels
    noisyPixels = int(noiseDensity * img.shape[0] * img.shape[1] / 100)
    #add the noise to the image
    noisyImg = img.copy()
    for i in range(noisyPixels):
        #add salt noise
        noisyImg[np.random.randint(0, img.shape[0]-1)][np.random.randint(0, img.shape[1]-1)] = 255
        #add pepper noise
        noisyImg[np.random.randint(0, img.shape[0]-1)][np.random.randint(0, img.shape[1]-1)] = 0
    
    return noisyImg

def movingAverageFilter(img, filterSize):
    # img is the input gray image
    # filterSize is the size of the filter
    # return the filtered image
    # The function uses moving average filter to remove the noise from the image   
    
    #apply the filter
    filteredImg = np.zeros_like(img)
    for i in range(len(img)-filterSize+1):
        for j in range(len(img[0]-filterSize+1)):
            filteredImg[i][j] = np.sum(img[i:i+filterSize, j:j+filterSize]/ filterSize**2)
    
    return filteredImg
    
def medianFilter(img, filterSize):
    # img is the input gray image
    # filterSize is the size of the filter
    # return the filtered image
    # The function uses median filter to remove the noise from the image   
           
    #apply the filter
    filteredImg = np.zeros_like(img)
    for i in range(len(img)):
        for j in range(len(img[0])):
            filteredImg[i][j] = np.median(img[i:i+filterSize, j:j+filterSize])
    
    return filteredImg

img = np.asarray(Image.open('flower.png'))
f, imgplot = plt.subplots(3,4)

imgplot[0][0].imshow(img, cmap='gray')

#add noise of noise to signal ratio 15dB
noisyImg = addGaussianNoise(img, 15)
imgplot[0][1].imshow(noisyImg, cmap='gray')

#add salt and pepper noise of density 25%
noisyImgSnP = addSaltAndPepperNoise(img, 25)
imgplot[0][2].imshow(noisyImgSnP, cmap='gray')

#apply moving average filter of size 3x3
filteredImgMovAv3x3 = movingAverageFilter(noisyImg, 3)
imgplot[1][0].imshow(filteredImgMovAv3x3, cmap='gray')
#apply moving average filter of size 11x11
filteredImgMovAv11x11 = movingAverageFilter(noisyImg, 11)
imgplot[1][1].imshow(filteredImgMovAv11x11, cmap='gray')

#apply moving average filter of size 3x3
filteredImgMovAv3x3SnP = movingAverageFilter(noisyImgSnP, 3)
imgplot[1][2].imshow(filteredImgMovAv3x3SnP, cmap='gray')
#apply moving average filter of size 11x11
filteredImgMovAv11x11SnP = movingAverageFilter(noisyImgSnP, 11)
imgplot[1][3].imshow(filteredImgMovAv11x11SnP, cmap='gray')

#apply median filter of size 3x3
filteredImgMedian3x3 = medianFilter(noisyImg, 3)
imgplot[2][0].imshow(filteredImgMedian3x3, cmap='gray')
#apply median filter of size 11x11
filteredImgMedian11x11 = medianFilter(noisyImg, 11)
imgplot[2][1].imshow(filteredImgMedian11x11, cmap='gray')

#apply median filter of size 3x3
filteredImgMedian3x3SnP = medianFilter(noisyImgSnP, 3)
imgplot[2][2].imshow(filteredImgMedian3x3SnP, cmap='gray')
#apply median filter of size 11x11
filteredImgMedian11x11SnP = medianFilter(noisyImgSnP, 11)
imgplot[2][3].imshow(filteredImgMedian11x11SnP, cmap='gray')

plt.show()