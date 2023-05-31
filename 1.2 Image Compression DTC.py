import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math
import cmath
import cv2

def twoDDCT(img, blockSize):
    # img is the input gray image
    # blockSize is the size of the block
    # return the DCT coefficients of the image
    dctImg = np.zeros_like(img)
    for i in range(0, len(img), blockSize):
        for j in range(0, len(img[0]), blockSize):
            dctImg[i:i+blockSize, j:j+blockSize] = cv2.dct(img[i:i+blockSize, j:j+blockSize])
             
    return dctImg

def inverseTwoDDCT(dctImg, blockSize):
    # dctImg is the DCT coefficients of the image
    # blockSize is the size of the block
    # return the reconstructed image from the DCT coefficients
    img = np.zeros_like(dctImg)
    for i in range(0, len(dctImg), blockSize):
        for j in range(0, len(dctImg[0]), blockSize):
            img[i:i+blockSize, j:j+blockSize] = cv2.idct(dctImg[i:i+blockSize, j:j+blockSize])
    return img

def zonalMask(blockSize, maskSize):
    # blockSize is the size of the block
    # maskSize is the size of the mask
    # return the zonal mask
    mask = np.zeros((blockSize, blockSize))
    for i in range(maskSize):
        for j in range(maskSize):
            if j-i >= 0:
                mask[i][j-i] = 1
    return mask

def thresholdingMask(blockSize, img, p):
    # blockSize is the size of the block
    # img is the input gray image
    # returns the masked image using thresholding
    # The function uses the n-largest coding in order to calculate the thressholding mask
    p = 1/p
    mask = np.zeros((blockSize, blockSize))
    thresholdingDCT = np.zeros_like(img)
    for i in range(0, len(img), blockSize):
        for j in range(0, len(img[0]), blockSize):
            block = img[i:i+blockSize, j:j+blockSize]
            sortedBlock = np.sort(block, axis=None)
            threshold = sortedBlock[-(blockSize//int(p))]
            for k in range(blockSize):
                for l in range(blockSize):
                    if block[k][l] >= threshold:
                        mask[k][l] = 1
            thresholdingDCT[i:i+blockSize, j:j+blockSize] = np.multiply(block, mask)
            
    return thresholdingDCT

def minSquareError(img1, img2):
    # img1 is the input gray image
    # img2 is the reconstructed image
    # return the MSE between the two images
    return np.mean((img1 - img2) ** 2)

img = cv2.imread("lenna.jpg", cv2.IMREAD_GRAYSCALE)
print(repr(img))
f, imgplot = plt.subplots(3,4)
imgplot[0][0].imshow(img, cmap='gray')

blockSize = 32
dctImg = twoDDCT(np.float32(img) / 255.0, blockSize)
imgplot[0][1].imshow(dctImg, cmap='gray')


zonalDCT = np.zeros_like(dctImg)
for i in range(0, len(dctImg), blockSize):
        for j in range(0, len(dctImg[0]), blockSize):
            zonalDCT[i:i+blockSize, j:j+blockSize] = np.multiply(dctImg[i:i+blockSize, j:j+blockSize], zonalMask(blockSize, blockSize))
    

imgplot[1][0].imshow(zonalDCT, cmap='gray')
imgplot[1][1].imshow(inverseTwoDDCT(zonalDCT, blockSize), cmap='gray')
zoneError = minSquareError(img, inverseTwoDDCT(zonalDCT, blockSize))
print(zoneError)
# imgplot[1][2].imshow(zoneError.clip(0,255).astype(np.uint8), cmap='gray')

zonalDCT2 = np.zeros_like(dctImg)
for i in range(0, len(dctImg), blockSize):
        for j in range(0, len(dctImg[0]), blockSize):
            zonalDCT2[i:i+blockSize, j:j+blockSize] = np.multiply(dctImg[i:i+blockSize, j:j+blockSize], zonalMask(blockSize, blockSize//10))
    

imgplot[1][2].imshow(zonalDCT2, cmap='gray')
imgplot[1][3].imshow(inverseTwoDDCT(zonalDCT2, blockSize), cmap='gray')

thresholdingDCT = thresholdingMask(blockSize, dctImg, 0.5)
imgplot[2][0].imshow(thresholdingDCT, cmap='gray')
imgplot[2][1].imshow(inverseTwoDDCT(thresholdingDCT, blockSize), cmap='gray')

thresholdingDCT2 = thresholdingMask(blockSize, dctImg, 0.05)
imgplot[2][2].imshow(thresholdingDCT2, cmap='gray')
imgplot[2][3].imshow(inverseTwoDDCT(thresholdingDCT2, blockSize), cmap='gray')

zonalError = []
for i in range(10, 1, -1):
    zonalDCT = np.zeros_like(dctImg)
    for j in range(0, len(dctImg), blockSize):
            for k in range(0, len(dctImg[0]), blockSize):
                zonalDCT[j:j+blockSize, k:k+blockSize] = np.multiply(dctImg[j:j+blockSize, k:k+blockSize], zonalMask(blockSize, blockSize//i))
    zonalError.append(minSquareError(img, inverseTwoDDCT(zonalDCT, blockSize)))

imgplot[0][2].plot(zonalError)

threholdError = []
for i in range(5, 50, 5):
    thresholdingDCT = thresholdingMask(blockSize, dctImg, i/100)
    threholdError.append(minSquareError(img, inverseTwoDDCT(thresholdingDCT, blockSize)))

imgplot[0][3].plot(threholdError)

plt.show()

