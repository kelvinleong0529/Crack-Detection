import cv2
import os

imageA = cv2.imread("fusion14_1.jpg")
imageB = cv2.imread("fusion16_2.jpg")

imageC = cv2.imread('fusion6_2.jpg')
imageD = cv2.imread("fusion6_3.jpg")

stitcher = cv2.Stitcher.create(0)
result1 = list(stitcher.stitch((imageA,imageB)))

result2 = list(stitcher.stitch((imageC,imageD)))
result = list(stitcher.stitch((result1[1],result2[1])))

result1[1] = cv2.resize(result1[1], (960, 540))
cv2.imwrite("result_fusion16.jpg", result1[1])
cv2.imshow("Results_fusion", result1[1])
cv2.imwrite('Stitching/built-inresult(3,4)'+'.jpg', result[1])
cv2.waitKey(0)
cv2.destroyAllWindows()