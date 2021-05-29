#Horizontal integral projection
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img=cv.imread("Extracting_info/largest_contour.jpg",0)
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

#Vertical Integral Projection
ret,img2=cv.threshold(img,80,255,cv.THRESH_BINARY)

#return the image height and width
(h,w)=img2.shape

# initialize a list with length equivalent to the image width, to record the number of dark pixels at every column
a =[0 for z in range(0,w)]

for i in range(0,w):           # traverse each column
    for j in range(0,h):       # traverse each row
        if img2[j,i]==0:       # decide whether it is a dark pixel, 0 meaning it is a dark pixel
            a[i]+=1            # increment the counter for that column
            img2[j,i]=255      # transform it into a white pixel after recording (by changing it into 255)
for i in range(0,w):           # traverse every column
    for j in range(h-a[i],h):  
        img2[j,i]=0            # transform it into a dark pixel

gridsize = (4, 4)
fig = plt.figure(figsize=(12, 6))
ax1 = plt.subplot2grid(gridsize, (0, 1), colspan=3, rowspan=3) 
ax1.imshow(img,cmap = 'gray')
plt.xticks([])
plt.yticks([])
ax2 = plt.subplot2grid(gridsize, (0, 0), rowspan=3)
img1 = cv.resize(img1,(500,1300))
ax2.imshow(img1,cmap = 'gray')
plt.xticks([])
plt.yticks([])
ax3 = plt.subplot2grid(gridsize, (3, 1), colspan=3)
img2 = cv.resize(img2,(3200,500))
ax3.imshow(img2,cmap = 'gray')
plt.xticks([])
plt.yticks([])
plt.show()