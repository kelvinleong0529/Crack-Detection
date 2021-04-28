import cv2
import os
import numpy as np

imageA = cv2.imread("group3_1.jpg")
imageB = cv2.imread("group3_2.jpg")
imageC = cv2.imread("group3_3.jpg")

bf = cv2.BFMatcher()
sift = cv2.xfeatures2d.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(imageA, None)
keypoints2, descriptors2 = sift.detectAndCompute(imageB, None)
keypoints3, descriptors3 = sift.detectAndCompute(imageC, None)
# Find matching points
matches = bf.knnMatch (descriptors1, descriptors2,k=2)

def draw_matches(img1, keypoints1, img2, keypoints2, matches):
    r, c = img1.shape[:2]
    r1, c1 = img2.shape[:2]

    # Create a blank image with the size of the first image + second image
    output_img = np.zeros((max([r, r1]), c+c1, 3), dtype='uint8')
    output_img[:r, :c, :] = np.dstack([img1, img1, img1])
    output_img[:r1, c:c+c1, :] = np.dstack([img2, img2, img2])

    # Go over all of the matching points and extract them
    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        (x1, y1) = keypoints1[img1_idx].pt
        (x2, y2) = keypoints2[img2_idx].pt

        # Draw circles on the keypoints
        cv2.circle(output_img, (int(x1),int(y1)), 4, (0, 255, 255), 1)
        cv2.circle(output_img, (int(x2)+c,int(y2)), 4, (0, 255, 255), 1)

        # Connect the same keypoints
        cv2.line(output_img, (int(x1),int(y1)), (int(x2)+c,int(y2)), (0, 255, 255), 1)
    
    return output_img

all_matches = []
for m, n in matches:
  # try to change [m] to m if fail!!!!!
  all_matches.append(m) 

good_matches = []
for m1, m2 in matches:
  if m1.distance < 0.6*m2.distance:
    good_matches.append(m1)

def warpImages(img1, img2, H):

  rows1, cols1 = img1.shape[:2]
  rows2, cols2 = img2.shape[:2]

  list_of_points_1 = np.float32([[0,0], [0, rows1],[cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
  temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)

  # When we have established a homography we need to warp perspective
  # Change field of view
  list_of_points_2 = cv2.perspectiveTransform(temp_points, H)

  list_of_points = np.concatenate((list_of_points_1,list_of_points_2), axis=0)

  [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
  [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
  
  translation_dist = [-x_min,-y_min]
  
  H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

  output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
  output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1

  return output_img

MIN_MATCH_COUNT = 10

if len(good_matches) > MIN_MATCH_COUNT:
    # Convert keypoints to an argument for findHomography
    src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    # Establish a homography
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    
    result1 = warpImages(imageB, imageA, M)

keypoints4, descriptors4 = sift.detectAndCompute(result1, None)
matches2 = bf.knnMatch (descriptors3, descriptors4,k=2)

all_matches2 = []
for m, n in matches2:
  all_matches2.append(m) #try to change [m] to m if fail!!!!!
print(len(all_matches2))

good_matches2 = []
for m1, m2 in matches2:
  if m1.distance < 0.6*m2.distance:
    good_matches.append(m1)

if len(all_matches2) > MIN_MATCH_COUNT:
    # Convert keypoints to an argument for findHomography
    src_pts = np.float32([ keypoints3[m.queryIdx].pt for m in all_matches2]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints4[m.trainIdx].pt for m in all_matches2]).reshape(-1,1,2)

    # Establish a homography
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    
    result2 = warpImages(result1, imageC, M)
    result2 = cv2.resize(result2, (960, 320))
    cv2.imwrite("result_SIFTmatch_final.jpg", result2)
    cv2.imshow("frame",result2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
