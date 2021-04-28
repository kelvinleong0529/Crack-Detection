import cv2 
import numpy as np
 
def sift_kp(image):
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sift=cv2.xfeatures2d.SIFT_create()
    kp,des = sift.detectAndCompute(image,None)
    kp_image = cv2.drawKeypoints(gray_image,kp,None)
    return kp_image,kp,des
 
 
def get_good_match(des1,des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2) #des1为模板图，des2为匹配图
    matches = sorted(matches,key=lambda x:x[0].distance/x[1].distance)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good
 
 
def siftImageAlignment(img1,img2):
   _,kp1,des1 = sift_kp(img1)
   _,kp2,des2 = sift_kp(img2)
   goodMatch = get_good_match(des1,des2) 
   if len(goodMatch) > 4:
       ptsA= np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
       ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
       ransacReprojThreshold = 4
       H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold);
       imgOut = cv2.warpPerspective(img2, H, (img1.shape[1],img1.shape[0]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
   return imgOut,H,status
 
img1 = cv2.imread('fusion1_2.jpg') 
img2 = cv2.imread('fusion1_3.jpg') 
 
_,kp1,des1 = sift_kp(img1)
_,kp2,des2 = sift_kp(img2)
goodMatch = get_good_match(des1,des2)
 
img3 = cv2.drawMatches(img1, kp1, img2, kp2, goodMatch[:5], None, flags=2)
#----or----
#goodMatch = np.expand_dims(goodMatch,1)
#img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, goodMatch[:5], None, flags=2)
 
img3 = cv2.resize(img3, (960, 540))
cv2.imshow('img',img3) 
cv2.waitKey(0) 
