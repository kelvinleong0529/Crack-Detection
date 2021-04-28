import cv2
import os
import numpy as np

class Stitcher:
    # stitcher function
    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        # load images
        (imageB, imageA) = images
        # detect A & B keypoints and calculate the corresponding descriptors
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # match these corresponding keypoints and return the results
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # if the result is None meaning there is no matching keypoints, exit current algo
        if M is None:
            return None

        # if result is !None, extract the matching results
        # H is a 3-by-3 angle transformation matrix
        (matches, H, status) = M
        # apply transformation on A, result is the corresponding transformed picture
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        # Pass B to the leftmost end of the result picture
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # Detect whether displaying matched image is needed
        if showMatches:
            # Generate the corresponding matching results
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            return (result, vis)

        # return the result
        return result

    def detectAndDescribe(self, image):
        # convert colour image to gray image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # initialize a SIFT generator
        descriptor = cv2.xfeatures2d.SIFT_create()
        # detect SIFT keypoints and calculate the corresponding descriptors
        (kps, features) = descriptor.detectAndCompute(gray, None)

        # convert the result into a NumPy array
        kps = np.float32([kp.pt for kp in kps])

        # return the keypoint array and corresponding descriptors 
        return kps, features

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # initialize brute-force matcher
        matcher = cv2.DescriptorMatcher_create("BruteForce")

        # match SIFT keypoints using KNN, 2 is the value of K
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

        matches = []
        for m in rawMatches:
            # When the ratio of the nearest distance to the next nearest 
            # distance is less than the ratio value, the matching pair is retained
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                # Store the index values of the 2 points in feature A and feature B
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # When the filtered matching pair > 4, the angle transformation matrix is calculated
        if len(matches) > 4:
            # Obtain the coordinates of the matching points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # Calculate the angle transformation matrix
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            return (matches, H, status)

        # if matching pair < 4, return None
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # Initialize the result image, then stitch A and B together
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # Draw the matching pairs
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # Display on the result image when the pairing is sucessfull
            if s == 1:
                # Draw the pairing results
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        return vis


imageA = cv2.imread('group5_1.jpg')
imageB = cv2.imread("group5_2.jpg")

imageC = cv2.imread('group5_2.jpg')
imageD = cv2.imread("group5_3.jpg")

# Combine the images into panaroma
stitcher = Stitcher()
(result1, vis1) = stitcher.stitch([imageA, imageB], showMatches=True)
(result2, vis2) = stitcher.stitch([imageC, imageD], showMatches=True)
(result3, vis3) = stitcher.stitch([result1, result2], showMatches=True)

vis3 = cv2.resize(vis3, (960, 540))
result3 = cv2.resize(result3, (960, 540))
cv2.imshow("Keypoint Matches", vis1)
cv2.imshow("Keypoint Matches", vis2)
cv2.imshow("Keypoint Matches", vis3)
cv2.imshow("Result", result3)
cv2.imwrite('result5_line'+'.jpg', vis3)
cv2.imwrite('result5'+'.jpg', result3)
cv2.waitKey(0)
cv2.destroyAllWindows()
