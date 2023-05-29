import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def histogram(img):
    # img is the input gray image
    # return the histogram of the image
    h = np.zeros(256)
    for i in range(len(img)):
        for j in range(len(img[0])):
            h[img[i][j]] += 1
    return h

def histogramEqualization(img):
    # img is the input gray image
    # return the histogram equalized image
    h = histogram(img)
    c = np.zeros(256)
    c[0] = h[0]
    for i in range(1, 256):
        c[i] = c[i-1] + h[i]
    c = c/(len(img)*len(img[0]))
    c = np.round(c*255)
    imgEq = np.zeros_like(img)
    for i in range(len(img)):
        for j in range(len(img[0])):
            imgEq[i][j] = c[img[i][j]]
    return imgEq

def localHistogramEqualization(img, blockSize):
    # img is the input gray image
    # blockSize is the size of the block
    # return the local histogram equalized image
    imgEq = np.zeros_like(img)
    for i in range(len(img) - blockSize):
        for j in range(len(img[0]) - blockSize):
            block = img[i:i+blockSize, j:j+blockSize]
            blockEq = histogramEqualization(block)
            imgEq[i:i+blockSize, j:j+blockSize] = blockEq
    return imgEq

img1 = np.asarray(Image.open('dark_road_1.jpg'))
img2 = np.asarray(Image.open('dark_road_2.jpg'))
img3 = np.asarray(Image.open('dark_road_3.jpg'))
# print(repr(img))

f, imgplot = plt.subplots(6,3)

imgplot[0][0].imshow(img1, cmap='gray')
imgplot[0][1].imshow(img2, cmap='gray')
imgplot[0][2].imshow(img3, cmap='gray')

h1 = histogram(img1)
h2 = histogram(img2)
h3 = histogram(img3)

imgplot[1][0].plot(h1)
imgplot[1][1].plot(h2)
imgplot[1][2].plot(h3)

imgEq1 = histogramEqualization(img1)
imgEq2 = histogramEqualization(img2)
imgEq3 = histogramEqualization(img3)

imgplot[2][0].imshow(imgEq1, cmap='gray')
imgplot[2][1].imshow(imgEq2, cmap='gray')
imgplot[2][2].imshow(imgEq3, cmap='gray')

h1Eq = histogram(imgEq1)
h2Eq = histogram(imgEq2)
h3Eq = histogram(imgEq3)

imgplot[3][0].plot(h1Eq)
imgplot[3][1].plot(h2Eq)
imgplot[3][2].plot(h3Eq)

blockSize = 4

imgLocEq1 = localHistogramEqualization(img1, 16)
imgLocEq2 = localHistogramEqualization(img2, 50)
imgLocEq3 = localHistogramEqualization(img3, 100)

imgplot[4][0].imshow(imgLocEq1, cmap='gray')
imgplot[4][1].imshow(imgLocEq2, cmap='gray')
imgplot[4][2].imshow(imgLocEq3, cmap='gray')

h1LocEq = histogram(imgLocEq1)
h2LocEq = histogram(imgLocEq2)
h3LocEq = histogram(imgLocEq3)

imgplot[5][0].plot(h1LocEq)
imgplot[5][1].plot(h2LocEq)
imgplot[5][2].plot(h3LocEq)

plt.show()