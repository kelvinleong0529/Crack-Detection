#Horizontal integral projection
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img=cv.imread("Extracting_info/morphological.jpg",0)
ret,img1=cv.threshold(img,80,255,cv.THRESH_BINARY)

#to get the height and width of image
(h,w)=img1.shape

#初始化一个跟图像高一样长度的数组，用于记录每一行的黑点个数
a=[0 for z in range(0,h)]

for i in range(0,h):          #遍历每一行
    for j in range(0,w):      #遍历每一列
        if img1[i,j]==0:      #判断该点是否为黑点，0代表黑点
            a[i]+=1           #该行的计数器加一
            img1[i,j]=255     #将其改为白点，即等于255
for i in range(0,h):          #遍历每一行
    for j in range(0,a[i]):   #从该行应该变黑的最左边的点开始向最右边的点设置黑点
        img1[i,j]=0           #设置黑点

#垂直投影
import numpy as np
import cv2 as cv
img=cv.imread("Extracting_info/morphological.jpg",0)
ret,img2=cv.threshold(img,80,255,cv.THRESH_BINARY)

#返回图像的高和宽
(h,w)=img2.shape

#初始化一个跟图像宽一样长度的数组，用于记录每一列的黑点个数
a =[0 for z in range(0,w)]

for i in range(0,w):           #遍历每一列
    for j in range(0,h):       #遍历每一行
        if img2[j,i]==0:       #判断该点是否为黑点，0代表是黑点
            a[i]+=1            #该列的计数器加1
            img2[j,i]=255      #记录完后将其变为白色，即等于255
for i in range(0,w):           #遍历每一列
    for j in range(h-a[i],h):  #从该列应该变黑的最顶部的开始向最底部设为黑点
        img2[j,i]=0            #设为黑点

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