# Import packages
# import numpy as np
import cv2
import image_converter
# import utils


def main():
    # Setting PATH for image(.jpg format)
    root_path = "images/"  # Root Path
    image_name = "1.jpg"  # Image Name
    image = root_path + image_name  # Path to image

    # JPG image to RGB format
    rgb_image_matrix = cv2.imread(image)

    # RESIZING THE RGB IMAGE
    resized_image_matrix = image_converter.resize_image(rgb_image_matrix, 640, 480)

    # CONVERTING RGB IMAGE TO GRAY SCALE IMAGE
    gray_scale_image_matrix = image_converter.rgb_to_gray(resized_image_matrix)

    # GIVING BLACK BORDER OF 1PX
    gray_scale_image_matrix = image_converter.give_black_border(gray_scale_image_matrix)

    # APPLYING BLUR EFFECT TO GRAY SCALE IMAGE
    blurred_image_matrix = image_converter.apply_blur_effects_to(gray_scale_image_matrix, effect="gaussian")

    cv2.imwrite("scanned/1_scanned" + ".jpg", resized_image_matrix)
    cv2.imwrite("scanned/2_scanned" + ".jpg", gray_scale_image_matrix)
    cv2.imwrite("scanned/3_scanned" + ".jpg", blurred_image_matrix)


if __name__ == '__main__':
    main()