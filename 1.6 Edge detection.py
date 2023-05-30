import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sys
from skimage import color 
import math
import cv2

img = np.asarray(Image.open('clock.jpg'))
imgGray = color.rgb2gray(img)
print(imgGray)

f, imgplot = plt.subplots(1,4)
imgplot[0].imshow(imgGray, cmap='gray')

def sobel(img):

    for i in range(len(img)-2):
        for j in range(len(img[0])-2):
            Gx=((2*img[i+2][j+1]+img[i+2][j]+img[i+2][j+2])-(2*img[i][j+1]+img[i][j]+img[i][j+2]))

            Gy=((2*img[i+1][j+2]+img[i][j+2]+img[i+2][j+2])-(2*img[i+1][j]+img[i][j]+img[i+2][j]))
            img[i][j]=math.sqrt(Gx**2+Gy**2)
    return img

def sobelThreshold(img, threshold):
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] < threshold:
                img[i][j] = 0
            else:
                img[i][j] = 255
    return img

def houghTransformation(img):
    # img is the input binary image
    # return the hough transformation result

    return img 

imgSobel = sobel(imgGray)
imgplot[1].imshow(imgSobel, cmap='gray')

imgSobelThreshold = sobelThreshold(imgSobel, np.mean(imgSobel))
imgplot[2].imshow(imgSobelThreshold, cmap='gray')

gray2rgb = cv2.cvtColor(imgSobelThreshold.astype(np.uint8),cv2.COLOR_GRAY2RGB)
gray = cv2.cvtColor(gray2rgb,cv2.COLOR_RGB2GRAY)
lines = cv2.HoughLines(gray,1,np.pi/180,2)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

imgplot[3].imshow(img, cmap='gray')


plt.show()