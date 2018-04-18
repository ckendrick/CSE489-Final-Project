import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('cat.png',0)
edges = cv2.Canny(img,32,32)
plt.subplot(121),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()