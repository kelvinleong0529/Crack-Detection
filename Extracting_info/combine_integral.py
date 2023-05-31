#Horizontal integral projection
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def preprocess_image(image_path, threshold_value):
    img = cv.imread(image_path, 0)
    _, img_binary = cv.threshold(img, threshold_value, 255, cv.THRESH_BINARY)
    return img_binary

def process_horizontal_projection(img_binary):
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

def process_vertical_projection(img_binary):
    h, w = img_binary.shape[:2]
    row_dark_pixel_counts = [0] * w

    for i in range(w):
        for j in range(h):
            if img_binary[j, i] == 0:
                row_dark_pixel_counts[i] += 1
                img_binary[j, i] = 255

    for i in range(w):
        for j in range(h - row_dark_pixel_counts[i], h):
            img_binary[j, i] = 0

    return img_binary

def show_images(img, img1, img2):
    gridsize = (4, 4)
    fig = plt.figure(figsize=(12, 6))

    ax1 = plt.subplot2grid(gridsize, (0, 1), colspan=3, rowspan=3)
    ax1.imshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])

    ax2 = plt.subplot2grid(gridsize, (0, 0), rowspan=3)
    img1 = cv.resize(img1, (500, 1300))
    ax2.imshow(img1, cmap='gray')
    plt.xticks([])
    plt.yticks([])

    ax3 = plt.subplot2grid(gridsize, (3, 1), colspan=3)
    img2 = cv.resize(img2, (3200, 500))
    ax3.imshow(img2, cmap='gray')
    plt.xticks([])
    plt.yticks([])

    plt.show()

def main():
    image_path = 'Extracting_info/largest_contour.jpg'
    threshold_value = 80

    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    img1 = preprocess_image(image_path, threshold_value)
    img2 = preprocess_image(image_path, threshold_value)

    img1 = process_horizontal_projection(img1)
    img2 = process_vertical_projection(img2)

    show_images(img, img1, img2)

if __name__ == '__main__':
    main()
