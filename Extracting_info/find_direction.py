import cv2
import numpy as np

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (4608, 3456))
    return image

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def process_contours(image, contours):
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)

        if area < 300:
            continue

        rect = cv2.minAreaRect(c)
        (pos, size, angle) = rect
        print("angle", angle * 180 / 3.142)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        rows, cols = image.shape[:2]
        [vx, vy, x, y] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
        lefty = int((-x * vy / vx) + y)
        righty = int(((cols - x) * vy / vx) + y)
        cv2.line(image, (cols - 1, righty), (0, lefty), (0, 0, 255), 2)
        #cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 1)

    return image

def show_image(image):
    cv2.imshow('image', image)
    print(image.shape)
    cv2.imwrite('Extracting_info/final_result.jpg', image)
    height, width, channels = image.shape
    #print(width)
    #print(height)
    cv2.waitKey()

def main():
    image_path = 'Extracting_info/largest_contour.jpg'
    image = load_image(image_path)
    thresh = preprocess_image(image)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = process_contours(image, contours)
    show_image(image_with_contours)

if __name__ == '__main__':
    main()
