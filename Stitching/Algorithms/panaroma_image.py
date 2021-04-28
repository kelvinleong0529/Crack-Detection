import sys
import cv2
import numpy as np

img = cv2.imread('group3_3.jpg')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

keypoints = sift.detect(img_gray, None)
result = cv2.resize(cv2.drawKeypoints(img, keypoints, None, (255, 0, 255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS),(950,540))
cv2.imshow("frame",result)
flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
cv2.imwrite("result_SIFT.jpg", result)
cv2.waitKey(0)
cv2.destroyAllWindows()