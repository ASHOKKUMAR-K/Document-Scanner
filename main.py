# Import packages
import numpy as np
import cv2
import image_converter
import utils


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

    # INITIALIZING THRESHOLD VALUE
    # threshold_1, threshold_2 = 20, 170

    # EDGE DETECTION
    sobel_kernel = image_converter.SobelKernel()
    gx, gy = sobel_kernel.find_gx_gy(blurred_image_matrix)  # Find GX and GY using Sobel Kernel

    canny_edge_detector = image_converter.CannyEdgeDetection()
    magnitude, angle = canny_edge_detector.find_magnitude_and_angle(gx, gy)
    image_after_non_max_suppression = canny_edge_detector.non_max_suppression(magnitude, angle)
    threshold_image, weak, strong = canny_edge_detector.threshold(image_after_non_max_suppression)
    cv2.imwrite("scanned/threshold_image" + ".jpg", threshold_image)
    canny_edge_image = canny_edge_detector.apply_hysteresis(threshold_image, weak, strong)

    # APPLY DILATION
    dilated_image = image_converter.apply_dilation(canny_edge_image, iteration=5)
    # APPLY EROSION
    eroded_image = image_converter.apply_erosion(dilated_image, iteration=1)

    cv2.imwrite("scanned/1_scanned" + ".jpg", resized_image_matrix)
    cv2.imwrite("scanned/2_scanned" + ".jpg", gray_scale_image_matrix)
    cv2.imwrite("scanned/3_scanned" + ".jpg", blurred_image_matrix)
    cv2.imwrite("scanned/gx_scanned" + ".jpg", gx)
    cv2.imwrite("scanned/gy_scanned" + ".jpg", gy)
    cv2.imwrite("scanned/magnitude_scanned" + ".jpg", magnitude)
    cv2.imwrite("scanned/angle_scanned" + ".jpg", angle)
    cv2.imwrite("scanned/image_after_non_max_suppression" + ".jpg", image_after_non_max_suppression)
    cv2.imwrite("scanned/canny_edge_image" + ".jpg", canny_edge_image)
    cv2.imwrite("scanned/dilated_image" + ".jpg", dilated_image)
    cv2.imwrite("scanned/eroded_image" + ".jpg", eroded_image)

if __name__ == '__main__':
    main()