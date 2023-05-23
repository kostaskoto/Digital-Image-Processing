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
            mask[i][j] = 1
    return mask

def thresholdingMask(blockSize, img):
    # blockSize is the size of the block
    # img is the input gray image
    # returns the masked image using thresholding
    # The function uses the n-largest coding in order to calculate the thressholding mask

    mask = np.zeros((blockSize, blockSize))
    thresholdingDCT = np.zeros_like(img)
    for i in range(0, len(img), blockSize):
        for j in range(0, len(img[0]), blockSize):
            block = img[i:i+blockSize, j:j+blockSize]
            sortedBlock = np.sort(block, axis=None)
            threshold = sortedBlock[-(blockSize//2)]
            for k in range(blockSize):
                for l in range(blockSize):
                    if block[k][l] >= threshold:
                        mask[k][l] = 1
            thresholdingDCT[i:i+blockSize, j:j+blockSize] = np.multiply(block, mask)
            
    return thresholdingDCT

img = cv2.imread("lenna.jpg", cv2.IMREAD_GRAYSCALE)
print(repr(img))
f, imgplot = plt.subplots(3,2)
imgplot[0][0].imshow(img, cmap='gray')

blockSize = 32
dctImg = twoDDCT(np.float32(img) / 255.0, blockSize)
imgplot[0][1].imshow(dctImg, cmap='gray')

zonalDCT = np.zeros_like(dctImg)
for i in range(0, len(dctImg), blockSize):
        for j in range(0, len(dctImg[0]), blockSize):
            zonalDCT[i:i+blockSize, j:j+blockSize] = np.multiply(dctImg[i:i+blockSize, j:j+blockSize], zonalMask(blockSize, 8))
    

imgplot[1][0].imshow(zonalDCT, cmap='gray')
imgplot[1][1].imshow(inverseTwoDDCT(zonalDCT, blockSize), cmap='gray')

thresholdingDCT = thresholdingMask(blockSize, dctImg)
imgplot[2][0].imshow(thresholdingDCT, cmap='gray')
imgplot[2][1].imshow(inverseTwoDDCT(thresholdingDCT, blockSize), cmap='gray')

plt.show()

