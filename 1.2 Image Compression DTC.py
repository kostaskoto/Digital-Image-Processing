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
            print(dctImg[i:i+blockSize, j:j+blockSize])
    

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

img = cv2.imread("lenna.jpg", cv2.IMREAD_GRAYSCALE)
print(repr(img))
f, imgplot = plt.subplots(2,3)
imgplot[0][0].imshow(img, cmap='gray')


dctImg = twoDDCT(np.float32(img) / 255.0, 32)
imgplot[0][1].imshow(dctImg, cmap='gray')

imgplot[0][2].imshow(inverseTwoDDCT(dctImg, 32), cmap='gray')

plt.show()

