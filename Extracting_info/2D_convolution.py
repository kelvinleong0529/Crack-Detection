import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('Extracting_info/cracks.jpg')
kernel = np.ones((5,5),np.float32)/25
dst = cv.filter2D(img,-1,kernel)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
cv.imwrite('Extracting_info/bilateral_result.jpg', dst)
plt.show()