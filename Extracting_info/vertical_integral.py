#Vertical Integral Projection
import numpy as np
import cv2 as cv
img=cv.imread("Extracting_info/morphological.jpg",0)
ret,img1=cv.threshold(img,80,255,cv.THRESH_BINARY)

#return the image height and width
(h,w)=img1.shape

# initialize a list with length equivalent to the image width, to record the number of dark pixels at every column
a =[0 for z in range(0,w)]

for i in range(0,w):           #traverse each column
    for j in range(0,h):       #traverse each row
        if img1[j,i]==0:       # decide whether it is a dark pixel, 0 meaning it is a dark pixel
            a[i]+=1            # increment the counter for that column
            img1[j,i]=255      # transform it into a white pixel after recording (by changing it into 255)
for i in range(0,w):           # traverse every column
    for j in range(h-a[i],h):  
        img1[j,i]=0            # transform it into a dark pixel
img1 = cv.resize(img1,(1000,500))
cv.imshow("img",img1)
cv.waitKey(0)
cv.imwrite('Extracting_info/vertical_integral.jpg', img1)
cv.destroyAllWindows()