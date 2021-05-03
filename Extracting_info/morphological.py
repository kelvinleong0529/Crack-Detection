import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('Extracting_info/canny_edge.jpg',0)
kernel = np.ones((3,3),np.uint8)
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

cv.imwrite('Extracting_info/morphological_new.jpg', closing)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(closing,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

