import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import random as rn

# Initialize the parameters for finding sub-pixel corners
# The maximum number of cycles allowed is 30, and the maximum error tolerance allowed is 0.001
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 20, 0.001)

# Obtain the position of the corner in the calibration plate
objp = np.zeros((6 * 9, 3), np.float32)

# The world coordinate system is built on the calibration board, all points will have coordinate values with 
# Z=0, we only need to assign values to X and Y
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Needed to store 3-Dimensional coordinates
obj_points = []

# Needed to store 2-Dimensional coordinates
img_points = []
projec_errors = []
sucess_count = 0

# Load callibration plate images
images = glob.glob("callibaration*.jpeg")
i=0
for fname in images:
    img = cv2.imread(fname)

    #convert the image from colour to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]

    # Locate the corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    print(corners)

    if ret:
        sucess_count += 1
        obj_points.append(objp)

        # find the sub-pixel accurate location
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        print(corners2)
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)

        # OpenCV drawing functions usually do not hav a return value
        cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        i+=1
        cv2.imwrite('Photos/conimg'+str(i)+'.jpg', img)
        cv2.waitKey(1500)

print(len(images))
cv2.destroyAllWindows()

# Assign the calibration results to different varaibles
# mtx = internal parameter matrix
# dist = distortion coefficient (k_1,k_2,p_1,p_2,k_3)
# rvecs = rotation vectors
# tvecs = translation vectors
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

print("ret:", ret)
print("内参数矩阵:\n", mtx) 
print("畸变系数:\n", dist)
print("旋转向量:\n", rvecs)
print("平移向量:\n", tvecs )

print("-----------------------------------------------------")

img = cv2.imread(images[23])
h, w = img.shape[:2]

# Display a wider range of image (certain area of the image will be deleted after normal remapping)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

#print (newcameramtx)
dst = cv2.undistort(img,mtx,dist,None,newcameramtx)

# Crop the image based on ROI(Region Of Interest)
x,y,w,h = roi
dst1 = dst[y:y+h,x:x+w]
print ("方法一:dst的大小为:", dst1.shape)
cv2.imwrite('calibresult.jpg', dst1)

# Calculation for reprojection error
total_error = 0
for i in range(len(obj_points)):
    imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(img_points[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    projec_errors.append(error)
    total_error += error
print ("total error: ", total_error/len(obj_points))

# plot the reprojection results using matplotlib
index = np.linspace(1,sucess_count,sucess_count,dtype = np.int32)
y = np.ones(1*sucess_count)*total_error/len(obj_points)
plt.plot(index,y,color="red",label="mean")
leg = plt.legend()
plt.bar(index, projec_errors)
plt.title('Mean error in pixels VS Image Index')
plt.xticks(np.arange(1,len(index)+1,1))
plt.xlabel('Image Index')
plt.ylabel('Mean error in pixels')
plt.show()