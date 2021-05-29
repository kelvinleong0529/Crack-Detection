import cv2
import numpy as np

# Load image, convert to grayscale, Otsu's threshold
image = cv2.imread('Extracting_info/largest_contour.jpg')
image = cv2.resize(image,(4608,3456))
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Find contours, obtain bounding rect, and draw width
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(type(cnts))
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c,True) # get the perimeter of the contour
    if area < 300:
        continue
    print(area)
    rect = cv2.minAreaRect(c)
    (pos,size,angle) = rect
    print("angle",angle*180/3.142)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    rows,cols = image.shape[:2]
    [vx,vy,x,y] = cv2.fitLine(c, cv2.DIST_L2,0,0.01,0.01)
    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    cv2.line(image,(cols-1,righty),(0,lefty),(0,0,255),2)
    #cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 1)

cv2.imshow('image', image)
print(image.shape)
cv2.imwrite('Extracting_info/final_result.jpg', image)
height, width, channels = image.shape
#print(width)
#print(height)
cv2.waitKey()
