#Horizontal Integral Projection
import numpy as np
import cv2 as cv
img=cv.imread("Extracting_info/morphological.jpg",0)
ret,img1=cv.threshold(img,80,255,cv.THRESH_BINARY)

#return the image height and width
(h,w)=img1.shape

# initialize a list with length equivalent to the image height, to record the number of dark pixels at every column
a=[0 for z in range(0,h)]

for i in range(0,h):          # traverse each row
    for j in range(0,w):      # traverse each column
        if img1[i,j]==0:      # decide whether it is a dark pixel, 0 meaning it is a dark pixel
            a[i]+=1           # increment the counter for that column
            img1[i,j]=255     # transform it into a white pixel after recording (by changing it into 255)
for i in range(0,h):          # traverse every row
    for j in range(0,a[i]):   
        img1[i,j]=0           # transform it into a dark pixel
img1 = cv.resize(img1,(1000,500))
cv.imshow("img",img1)
cv.imwrite('Extracting_info/horizontal_integral.jpg', img1)
cv.waitKey(0)
cv.destroyAllWindows()