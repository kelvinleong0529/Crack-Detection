import cv2
import numpy as np

# Load image, convert to grayscale, Otsu's threshold
image = cv2.imread('Extracting_info/largest_contour.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Find contours, obtain bounding rect, and draw width
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(type(cnts))
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    if area < 3:
        continue
    if h < 30 or h >150:
        continue
    print(area)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    print(box)
    image = cv2.drawContours(image,[box],0,(0,0,255),2)
    cv2.putText(image, str(area), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
    #cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 1)

cv2.imshow('image', image)
cv2.imwrite('Extracting_info/final_result.jpg', image)
cv2.waitKey()