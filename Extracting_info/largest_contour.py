import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = cv2.imread('Extracting_info/morphological_new.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #find contours of all the components and holes 
    gray_temp = gray.copy() #copy the gray image because function
                            #findContours will change the imput image into another  
    contours, hierarchy = cv2.findContours(gray_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #show the contours of the imput image
    cv2.drawContours(img, contours, -1, (255, 255, 255), 2)
    plt.figure('original image with contours'), plt.imshow(img, cmap = 'gray')

    #find the max area of all the contours and fill it with 0
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    max_idx = np.argmax(area)
    cv2.fillConvexPoly(gray, contours[max_idx], 0)
    #show image without max connect components 
    plt.figure('remove max connect com'), plt.imshow(gray, cmap = 'gray')

    #cv2.imwrite('Extracting_info/largest_contour.jpg', img)
    plt.show()
