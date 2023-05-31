import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def apply_filter(image_path, kernel):
    img = cv.imread(image_path)
    dst = cv.filter2D(img, -1, kernel)
    return dst

def show_images(original, filtered):
    plt.subplot(121), plt.imshow(original), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(filtered), plt.title('Filtered')
    plt.xticks([]), plt.yticks([])
    plt.show()

def save_image(image, output_path):
    cv.imwrite(output_path, image)

def main():
    img_path = 'Extracting_info/cracks.jpg'
    output_path = 'Extracting_info/bilateral_result.jpg'
    kernel = np.ones((5, 5), np.float32) / 25

    filtered_img = apply_filter(img_path, kernel)
    show_images(cv.imread(img_path), filtered_img)
    save_image(filtered_img, output_path)

if __name__ == '__main__':
    main()
