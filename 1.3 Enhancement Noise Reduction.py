import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

img = np.asarray(Image.open('flower.png'))
print(repr(img))
imgplot = plt.imshow(img, cmap='gray')
plt.show()