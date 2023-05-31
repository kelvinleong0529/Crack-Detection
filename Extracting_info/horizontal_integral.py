#Horizontal Integral Projection
import numpy as np
import cv2 as cv

def preprocess_image(image_path, threshold_value):
    img = cv.imread(image_path, 0)
    _, img_binary = cv.threshold(img, threshold_value, 255, cv.THRESH_BINARY)
    return img_binary

def calculate_column_dark_pixel_counts(img_binary):
    h, w = img_binary.shape[:2]
    column_dark_pixel_counts = [0] * h

    for i in range(h):
        for j in range(w):
            if img_binary[i, j] == 0:
                column_dark_pixel_counts[i] += 1
                img_binary[i, j] = 255

    for i in range(h):
        for j in range(column_dark_pixel_counts[i]):
            img_binary[i, j] = 0

    return img_binary

def show_image(image):
    resized_image = cv.resize(image, (1000, 500))
    cv.imshow("img", resized_image)
    cv.imwrite('Extracting_info/horizontal_integral.jpg', resized_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def main():
    image_path = 'Extracting_info/morphological.jpg'
    threshold_value = 80

    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    img_binary = preprocess_image(image_path, threshold_value)
    img_binary = calculate_column_dark_pixel_counts(img_binary)

    show_image(img_binary)

if __name__ == '__main__':
    main()
