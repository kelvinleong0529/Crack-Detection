import cv2 as cv
import numpy as np

if  __name__ == "__main__":
    img = cv.imread("fusion1_1.jpg") #LOGO
    img1 = cv.imread("fusion1_3.jpg") #MESSI
    # look(img)
    rows, cols, channels = img.shape
    roi = img1[0:rows, 0:cols] #获得messi的ROI
    img2gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)# 颜色空间的转换
    ret, mask = cv.threshold(img2gray, 20, 255, cv.THRESH_BINARY)# 掩码 黑色
    mask_inv = cv.bitwise_not(mask)# 掩码取反 白色
    # #取mask中像素不为的0的值，其余为0
    img1_bg = cv.bitwise_and(img, img, mask=mask)
    img2_fg = cv.bitwise_and(roi, roi, mask=mask_inv)
    dst = cv.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst
    img1 = cv.resize(img1, (960, 540))
    cv.imshow("result", img1)
    cv.waitKey()
    cv.destroyAllWindows()