import cv2
import numpy as np
import imutils
img = cv2.imread('Extracting_info/negative_image.png')
img00=np.uint8(np.log1p(img))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 150, 150)
edged = cv2.dilate(edged, None, iterations=2)
edged = cv2.erode(edged, None, iterations=1)
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

cv2.imshow("log",edged)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imwrite('Extracting_info/logarithmic_result.jpg', normalized_image)