import cv2
import matplotlib.pyplot as plt

img_bgr = cv2.imread('Extracting_info/new_modify.png', 1)

# get height and width of the image
height, width, _ = img_bgr.shape
  
for i in range(0, height - 1):
    for j in range(0, width - 1):
          
        # Get the pixel value
        pixel = img_bgr[i, j]
          
        # Negate each channel by 
        # subtracting it from 255
          
        # 1st index contains red pixel
        pixel[0] = 255 - pixel[0]
          
        # 2nd index contains green pixel
        pixel[1] = 255 - pixel[1]
          
        # 3rd index contains blue pixel
        pixel[2] = 255 - pixel[2]
          
        # Store new values in the pixel
        img_bgr[i, j] = pixel
  
# Display the negative transformed image
plt.imshow(img_bgr)
plt.show()
cv2.imwrite('Extracting_info/negative_image.png', img_bgr)