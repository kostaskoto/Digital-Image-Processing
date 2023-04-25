import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

img = np.asarray(Image.open('factory.jpg'))
print(repr(img))
imgplot = plt.imshow(img)
plt.show()