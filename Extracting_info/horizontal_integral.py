#水平投影
import numpy as np
import cv2 as cv
img=cv.imread("Extracting_info/morphological.jpg",0)
ret,img1=cv.threshold(img,80,255,cv.THRESH_BINARY)

#返回图像的高和宽
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
cv.imshow("img",img1)
cv.waitKey(0)
cv.destroyAllWindows()