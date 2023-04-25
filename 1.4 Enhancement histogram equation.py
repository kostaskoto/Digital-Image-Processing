import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

img1 = np.asarray(Image.open('dark_road_1.jpg'))
img2 = np.asarray(Image.open('dark_road_2.jpg'))
img3 = np.asarray(Image.open('dark_road_3.jpg'))
# print(repr(img))

f, imgplot = plt.subplots(1,3)

imgplot[0].imshow(img1, cmap='gray')
imgplot[1].imshow(img2, cmap='gray')
imgplot[2].imshow(img3, cmap='gray')
plt.show()