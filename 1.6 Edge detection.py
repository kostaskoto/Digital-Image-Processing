import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sys
from skimage import color 
import math

img = np.asarray(Image.open('clock.jpg'))
imgGray = color.rgb2gray(img)
print(imgGray)

f, imgplot = plt.subplots(1,2)
imgplot[0].imshow(imgGray, cmap='gray')

#ex 1
for i in range(len(imgGray)-2):
    for j in range(len(imgGray[0])-2):
        Gx=((2*imgGray[i+2][j+1]+imgGray[i+2][j]+imgGray[i+2][j+2])-(2*imgGray[i][j+1]+imgGray[i][j]+imgGray[i][j+2]))

        Gy=((2*imgGray[i+1][j+2]+imgGray[i][j+2]+imgGray[i+2][j+2])-(2*imgGray[i+1][j]+imgGray[i][j]+imgGray[i+2][j]))
        imgGray[i][j]=math.sqrt(Gx**2+Gy**2)

imgplot[1].imshow(imgGray, cmap='gray')

#ex2

plt.show()