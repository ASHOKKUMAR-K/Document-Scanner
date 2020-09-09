# Import packages
import cv2  # Used only to read and write images
import image_converter  # consists of methods needed for image conversion


class Scanner:
    """
    Scan Images
    """
    @staticmethod
    def scan(image):
        """
        Scans a single image uploaded.
        :param image: Image Path
        """
        # Setting PATH for image(.jpg format)
        # root_path = "images/"  # Root Path
        # image_name = "1.jpg"  # Image Name
        # image = root_path + image_name  # Path to image

        # JPG image to RGB format
        rgb_image_matrix = cv2.imread(image)

        # RESIZING THE RGB IMAGE
        resized_image_matrix = image_converter.resize_image(rgb_image_matrix, 640, 480)

        # CONVERTING RGB IMAGE TO GRAY SCALE IMAGE
        gray_scale_image_matrix = image_converter.rgb_to_gray(resized_image_matrix)

        # GIVING BLACK BORDER OF 1PX
        gray_scale_image_matrix = image_converter.give_black_border(gray_scale_image_matrix)

        # CANNY EDGE DETECTION
        # STEP 1: APPLYING GAUSSIAN BLUR EFFECT TO GRAY SCALE IMAGE TO REDUCE NOISE
        blurred_image_matrix = image_converter.apply_blur_effects_to(gray_scale_image_matrix,
                                                                     effect="gaussian")
        # STEP 2: FINDING INTENSITY GRADIENTS OF AN IMAGE
        # 1. Gx
        # 2. Gy
        # 3. Magnitude = sqrt( Gx^2 + Gy^2 )
        # 4. Angle = arctan2( Gy / Gx )
        sobel_kernel = image_converter.SobelKernel()  # Initializing Sobel Kernel
        gx, gy = sobel_kernel.find_gx_gy(blurred_image_matrix)  # Finding GX and GY using Sobel Kernel
        canny_edge_detector = image_converter.CannyEdgeDetection()  # Initializing Canny Edge Detector
        magnitude, angle = canny_edge_detector.find_magnitude_and_angle(gx, gy)  # Finding Magnitude and Angle
        # STEP 3: APPLYING NON-MAX SUPPRESSION TO THE SMOOTHENED IMAGE USING MAGNITUDE AND ANGLE
        image_after_non_max_suppression = canny_edge_detector.non_max_suppression(blurred_image_matrix,
                                                                                  magnitude, angle)
        # STEP 4: DOUBLE THRESHOLD
        threshold_image, weak, strong = canny_edge_detector.threshold(image_after_non_max_suppression)
        # STEP 5: EDGE TRACKING BY HYSTERESIS
        canny_edge_image = canny_edge_detector.apply_hysteresis(threshold_image, weak, strong)

        # APPLY DILATION
        dilated_image = image_converter.apply_dilation(canny_edge_image, iteration=5)

        # APPLY EROSION
        eroded_image = image_converter.apply_erosion(dilated_image, iteration=1)

        # CROPPING IMAGE
        crops = image_converter.crop_and_get_document(eroded_image)
        cropped_image = gray_scale_image_matrix[crops[0]:crops[1], crops[2]:crops[3]]

        # SCANNING CROPPED IMAGE
        # STEP 1: APPLYING ADAPTIVE THRESHOLD
        adaptive_threshold_image = image_converter.apply_adaptive_threshold(cropped_image)
        # STEP 2: CHANGING THRESHOLD VALUES (255 --> 1) AND (1 --> 255)
        scanned_image = image_converter.apply_bitwise_not(adaptive_threshold_image)
        # STEP 3: APPLYING MEDIAN BLUR
        # scanned_image = image_converter.apply_median_blur(scanned_image)

        # SAVING SCANNED IMAGE
        cv2.imwrite("static/scanned/final_scanned" + ".jpg", scanned_image)