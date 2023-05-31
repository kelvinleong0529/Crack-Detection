import cv2

def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return bw_img

def show_image(image):
    cv2.imshow("Binary Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    image_path = 'Extracting_info/negative_image.png'
    binary_image = process_image(image_path)
    show_image(binary_image)

if __name__ == '__main__':
    main()
