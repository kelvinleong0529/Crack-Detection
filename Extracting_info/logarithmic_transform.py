import cv2
import numpy as np
import matplotlib.pyplot as plt
   
# Read an image
image = cv2.imread('Extracting_info/new_modify.png')
   
# Apply log transformation method
c = 255 / np.log(1 + np.max(image))
log_image = c * (np.log(image + 1))
   
# Specify the data type so that
# float value will be converted to int
log_image = np.array(log_image, dtype = np.uint8)
   
# Display both images
plt.imshow(log_image)
plt.show()
cv2.imwrite('Extracting_info/logarithmic_result.jpg', log_image)
plt.show()