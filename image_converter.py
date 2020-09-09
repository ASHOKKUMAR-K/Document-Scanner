# Import packages
import utils
import numpy as np


def resize_image(image, to_height=640, to_width=480):
    """
    Resizes RGB Image to desired Height and Width

    :param image: RGB Image Matrix, type: numpy.ndarray
    :param to_height: Desired height, type: int
    :param to_width: Desired Width, type: int
    :return: Resized Image, type: numpy.ndarray
    """

    original_dimension = utils.dimension_of_matrix(image)  # Getting Original Dimensions of the RGB Image Matrix
    # Getting optimal dimensions for the desired height and width
    row_px, col_px, optimal_dimension = utils.choose_optimal_dimension(original_dimension, to_height, to_width)

    # Initialize Blank image of resize shape
    resized_image = np.zeros((optimal_dimension[0], optimal_dimension[1], 3), np.uint8)

    # i ==> Height ==> Row
    # j ==> Width ==> Column
    start_i = 0  # Start Row Pixel of original RGB image
    resize_i = 0  # Start Row Pixel of Resized RGB image
    while start_i <= optimal_dimension[0] * row_px:  # Iterate till start_i reach height of original Image
        start_j = 0  # Start Column Pixel of original RGB Image
        resize_j = 0  # Start Column Pixel of Resized RGB Image
        end_i = start_i + row_px  # End Row Pixel of original RGB image
        if end_i > optimal_dimension[0] * row_px:  # Break when crosses last row pixel of original Image
            break
        while start_j <= optimal_dimension[1] * col_px:  # Iterate till start_j reach width of original Image
            end_j = start_j + col_px  # End Column Pixel of original RGB image
            if end_j > optimal_dimension[1] * col_px:  # Break when crosses last column pixel of original Image
                break

            # Multiple pixel to Single pixel --> Replacing with average RGB pixel values
            resized_image[resize_i][resize_j] = reduce_to_one_px(image,
                                                                 start_i, start_j,
                                                                 end_i, end_j,
                                                                 row_px * col_px)
            resize_j += 1  # Incrementing Column Pixel for Resized image
            start_j += col_px  # Incrementing Column Pixel with number of column pixel considered for average
        resize_i += 1  # Incrementing Row Pixel for Resized image
        start_i += row_px  # Incrementing Column Pixel with number of row pixel considered for average

    return resized_image


def give_black_border(gray_image):
    """
    Add zeros on the border of the Gray Scale Image Matrix
    :param gray_image: Gray Scale Image Matrix, type: numpy.ndarray
    :return: Gray Scale Image Matrix with Zero Border, type: numpy.ndarray
    """
    rows = gray_image.shape[0]
    zeros = np.zeros((rows, 1), np.uint8)
    gray_image = np.hstack((gray_image, zeros))  # Add Zeros to the Right of the matrix
    gray_image = np.hstack((zeros, gray_image))  # Add Zeros to the Left of the matrix

    cols = gray_image.shape[1]
    zeros = np.zeros((1, cols), np.uint8)
    gray_image = np.vstack((gray_image, zeros))  # Add Zeros to the Bottom of te matrix
    gray_image = np.vstack((zeros, gray_image))  # Add Zeros to the Top of the matrix

    return gray_image


def reduce_to_one_px(image, start_i, start_j, end_i, end_j, total_px):
    """
    Convert group of pixels to a single pixel by taking average
    :param image: RGB Image Matrix, type: numpy.ndarray
    :param start_i: Start Row Index of Group of Pixels, type: int
    :param start_j: Start Column Index of Group of Pixels, type: int
    :param end_i: End Row Index of Group of Pixels, type: int
    :param end_j: End Column Index of Group of Pixels, type: int
    :param total_px: Total individual pixels in the group of pixels, type: int
    :return: Single Pixel of size 3(R, G, B), type: numpy.ndarray
    """
    pixel = np.array([0, 0, 0])  # Initializing with single Black coloured pixel
    for i in range(start_i, end_i, 1):
        for j in range(start_j, end_j, 1):
            pixel += image[i][j]

    pixel = pixel / total_px  # Computes Average
    pixel = np.array(pixel // 1, np.int8)  # Converting to int datatype
    return pixel


def rgb_to_gray(rgb_image):
    """
    Converts RGB Image to Gray Scale Image
    :param rgb_image: Resized RGB Image Matrix, type: numpy.ndarray
    :return: Gray Scale Image Matrix, type: numpy.ndarray
    """
    dimension = utils.dimension_of_matrix(rgb_image)  # Dimension of RGB Image
    # GRAY SCALE CONVERSION
    # GRAY = (Red + Green + Blue) / 3
    gray_scale = np.array([[np.sum(rgb_image[i][j] // 3) // 1 for j in range(dimension[1])]
                           for i in range(dimension[0])], np.uint8)
    return gray_scale


def apply_blur_effects_to(gray_image, effect="mean"):
    """
    Applies Smoothening Effects to Gray Scale Image
    :param gray_image: Gray Scale Image Matrix, type: numpy.ndarray
    :param effect: Type of Blur effects, type: str
    :return: Smoothened Image Matrix, type: numpy.ndarray
    """
    if effect == "mean":  # To apply Mean Blur
        return apply_mean_blur_effects_to(gray_image)
    elif effect == "gaussian":  # To apply Gaussian Blur
        return apply_gaussian_blur_effects_to(gray_image)


def apply_mean_blur_effects_to(gray_image):
    """
    Smoothens the image with Mean Blur
    :param gray_image: Gray Scale Image Matrix, type: numpy.ndarray
    :return: Smoothened Image Matrix
    """
    dimension = utils.dimension_of_matrix(gray_image)  # Dimension of Gray-scale Image
    blurred_image = np.copy(gray_image)  # Deep copy of Gray-scale Image

    for i in range(1, dimension[0] - 1):
        for j in range(1, dimension[1] - 1):
            # Smoothening each pixels of image by taking mean of 3 x 3 kernel
            blurred_image[i][j] = (np.sum(gray_image[i - 1:i + 2, j - 1:j + 2]) // 9) // 1
    return np.array(blurred_image, np.uint8)


def apply_gaussian_blur_effects_to(gray_image):
    """
    Smoothens the image with Gaussian Blur
    :param gray_image: Gray Scale Image Matrix, type: numpy.ndarray
    :return: Smoothened Image Matrix
    """
    dimension = utils.dimension_of_matrix(gray_image)  # Dimension of Image
    blurred_image = np.copy(gray_image)  # Copy of Gray-scale Image

    for i in range(1, dimension[0] - 1):
        for j in range(1, dimension[1] - 1):
            blurred_image[i][j] = gaussian(gray_image[i - 1:i + 2, j - 1:j + 2])
    return np.array(blurred_image, np.uint8)


def gaussian(matrix):
    """
    Returns Gaussian Blur value of the center pixel of the Matrix
    :param matrix: 3 x 3 Matrix of pixel, type: numpy.ndarray
    :return: Gaussian Blur Value
    """
    multiplier = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    pixel = 0
    # Computes Gaussian Blur Value
    for i in range(3):
        for j in range(3):
            pixel += (matrix[i][j] * multiplier[i][j]) // 16
    return np.round(pixel)


class SobelKernel:
    """
    SOBEL Kernel to find Gx and Gy
    """
    def find_gx_gy(self, smooth_image):
        """
        Returns the values of Gx and Gy from the smoothened Image
        :param smooth_image: Smoothened Image matrix, type: numpy.ndarray
        :return: Gx and Gy, type: tuple(int, int)
        """
        gx = self.find_gx(smooth_image)  # Gx
        gy = self.find_gy(smooth_image)  # Gy

        return gx, gy

    def find_gx(self, smooth_image):
        """
        Returns Horizontal Intensity Gradient Image Matrix
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]]
        :param smooth_image: Smoothened Image
        :return: Gx Matrix
        """
        dimension = smooth_image.shape
        multiplier = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        gx = np.zeros(dimension)
        for i in range(1, dimension[0] - 1):
            for j in range(1, dimension[1] - 1):
                # Horizontal Intensity Gradients of the pixel
                gx[i][j] = self.find_value(smooth_image[i-1:i+2, j-1:j+2], multiplier)
        return gx

    def find_gy(self, smooth_image):
        """
        Returns Vertical Intensity Gradient Image Matrix
        [[-1,  0,  1],
         [-2,  0,  2],
         [-1,  0,  1]]
        :param smooth_image:
        :return:
        """
        dimension = smooth_image.shape
        multiplier = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        gy = np.zeros(dimension)
        for i in range(1, dimension[0] - 1):
            for j in range(1, dimension[1] - 1):
                # Vertical Intensity Gradients of the pixel
                gy[i][j] = self.find_value(smooth_image[i-1:i+2, j-1:j+2], multiplier)
        return gy

    @staticmethod
    def find_value(a, b):
        return np.sum(a * b)


class CannyEdgeDetection:
    """
    Detects Edge of the image using Canny Edge Detector
    """
    def find_magnitude_and_angle(self, gx, gy):
        """
        Returns the magnitude and angle of the image.
        :param gx: Horizontal Intensity Gradient of the Matrix
        :param gy: Vertical Intensity Gradient of the Matrix
        :return: Magnitude and Angle of the matrix
        """
        dimension = gx.shape
        magnitude = np.zeros(dimension)
        angle = np.zeros(dimension)

        for i in range(dimension[0]):
            for j in range(dimension[1]):
                magnitude[i][j] = self.magnitude_(gx[i][j], gy[i][j])
                angle[i][j] = self.angle_(gx[i][j], gy[i][j])
        return magnitude, angle

    @staticmethod
    def magnitude_(gx, gy):
        """
        Magnitude of the Image
        :param gx: Horizontal Intensity Gradient of the Pixel
        :param gy: Vertical Intensity Gradient of the Pixel
        :return: Magnitude of the Pixel
        """
        return np.sqrt(gx**2 + gy**2)

    @staticmethod
    def angle_(gx, gy):
        """
        Angle of the Image in Radians
        :param gx: Horizontal Intensity Gradient of the Pixel
        :param gy: Vertical Intensity Gradient of the Pixel
        :return: Angle of the Pixel in Radians
        """
        return np.arctan2(gy, gx)

    @staticmethod
    def non_max_suppression(smooth_image, magnitude, angle):
        """
        Non-maximum suppression is an edge thinning technique.
        Non-maximum suppression is applied to find the locations
        with the sharpest change of intensity value.
        :param smooth_image: Smoothened Image
        :param magnitude: Magnitude Matrix
        :param angle: Angle of the Matrix in Radians
        :return: Sharpen the Intensity
        """
        dimension = magnitude.shape
        suppressed_image = np.copy(smooth_image)
        angle = (angle / np.pi) * 180  # Conversion from radians to Degrees
        angle[angle < 0] += 180

        for i in range(1, dimension[0] - 1):
            for j in range(1, dimension[1] - 1):
                try:
                    q = 255
                    r = 255
                    # angle 0 [157.5 to 22.5]
                    if (0 <= angle[i][j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = magnitude[i][j + 1]
                        r = magnitude[i][j - 1]
                    # angle 45
                    elif 22.5 <= angle[i, j] < 67.5:
                        q = magnitude[i + 1, j - 1]
                        r = magnitude[i - 1, j + 1]
                    # angle 90
                    elif 67.5 <= angle[i, j] < 112.5:
                        q = magnitude[i + 1, j]
                        r = magnitude[i - 1, j]
                    # angle 135
                    elif 112.5 <= angle[i, j] < 157.5:
                        q = magnitude[i - 1, j - 1]
                        r = magnitude[i + 1, j + 1]
                    else:
                        pass

                    if magnitude[i][j] >= q and magnitude[i][j] >= r:
                        suppressed_image[i][j] = smooth_image[i][j]
                    else:
                        suppressed_image[i][j] = 0
                except IndexError:
                    pass
        return suppressed_image

    @staticmethod
    def threshold(suppressed_image):
        """
        Returns image with only three possible pixel values zero, weak and strong
        :param suppressed_image: Image after non-max suppression
        :return: Threshold image with only three possible pixel values zero, weak and strong
        """
        high_threshold = np.max(suppressed_image) * 0.70  # Setting High Threshold Value
        low_threshold = high_threshold * 0.30  # Setting Low Threshold Value
        threshold_image = np.copy(suppressed_image)

        weak = np.int32(30)  # Value for weak pixels
        strong = np.int32(255)  # Value for strong pixels

        strong_i, strong_j = np.where(suppressed_image >= high_threshold)  # If pixel >= High Threshold
        zeros_i, zeros_j = np.where(suppressed_image < low_threshold)  # If pixel < Low Threshold
        # If Low Threshold <= Pixel < High Threshold
        weak_i, weak_j = np.where((suppressed_image < high_threshold) & (suppressed_image >= low_threshold))

        threshold_image[strong_i, strong_j] = strong
        threshold_image[weak_i, weak_j] = weak
        threshold_image[zeros_i, zeros_j] = 0

        return threshold_image, weak, strong

    @staticmethod
    def apply_hysteresis(threshold_image, weak, strong):
        """
        Tracking Edge by Hysteresis.
        Converts weak pixel to either 0 or 255 by means of neighbour pixel.
        :param threshold_image: Threshold image, type: numpy.ndarray
        :param weak: weak pixel value, type: int
        :param strong: strong pixel value, type: int
        :return: Image with only two possible pixel values (0 and 255)
        """
        dimension = threshold_image.shape
        for i in range(1, dimension[0] - 1):
            for j in range(1, dimension[1] - 1):
                if threshold_image[i][j] == weak:
                    try:
                        if (threshold_image[i-1, j-1] == strong or threshold_image[i-1, j] == strong or
                                threshold_image[i-1, j+1] == strong or threshold_image[i, j-1] == strong or
                                threshold_image[i, j+1] == strong or threshold_image[i+1, j-1] == strong or
                                threshold_image[i+1, j] == strong or threshold_image[i+1, j+1] == strong):
                            threshold_image[i][j] = strong
                        else:
                            threshold_image[i][j] = 0
                    except IndexError:
                        pass
        return threshold_image


def apply_dilation(canny_edge_image, iteration=3):
    """
    Applies dilation to the Image
    :param canny_edge_image: Edge detected image, type: numpy.ndarray
    :param iteration: Number of iteration to apply dilation, type: int
    :return: Dilated Image
    """
    dimension = canny_edge_image.shape
    dilated_image = np.copy(canny_edge_image)

    for _ in range(iteration):
        for i in range(1, dimension[0] - 1):
            for j in range(1, dimension[1] - 1):
                if canny_edge_image[i][j] == 0:
                    if (canny_edge_image[i-1][j] == 255 or canny_edge_image[i+1][j] == 255 or
                            canny_edge_image[i][j-1] == 255 or canny_edge_image[i][j+1] == 255):
                        dilated_image[i][j] = 255
                    else:
                        dilated_image[i][j] = 0
        canny_edge_image = np.copy(dilated_image)

    return dilated_image


def apply_erosion(dilated_image, iteration=1):
    dimension = dilated_image.shape
    eroded_image = np.copy(dilated_image)

    for _ in range(iteration):
        for i in range(1, dimension[0] - 1):
            for j in range(1, dimension[1] - 1):
                if dilated_image[i][j] == 255:
                    if (dilated_image[i-1][j] == 255 and dilated_image[i+1][j] == 255
                            and dilated_image[i][j-1] == 255 and dilated_image[i][j+1] == 255):
                        eroded_image[i][j] = 255
                    else:
                        eroded_image[i][j] = 0
        dilated_image = np.copy(eroded_image)
    return eroded_image


def crop_and_get_document(image):
    """
    Detects the four edges of the images.
    Top row, Botom row, Left column, Right column
    :param image: Image Matrix
    :return: Cropping row and column.
    """
    dimension = image.shape
    # Center Pixel of the image
    center_i, center_j = np.int32(dimension[0] // 2), np.int32(dimension[1] // 2)

    # Top Left Pixel
    top_left_i, top_left_j = center_i, center_j
    while True:
        if image[top_left_i - 10, top_left_j - 10] == 255:
            top_left_i, top_left_j = top_left_i - 10, top_left_j - 10
        elif image[top_left_i - 10, top_left_j] == 255:
            top_left_i, top_left_j = top_left_i - 10, top_left_j
        elif image[top_left_i, top_left_j - 10] == 255:
            top_left_i, top_left_j = top_left_i, top_left_j - 10

        elif image[top_left_i - 5, top_left_j - 5] == 255:
            top_left_i, top_left_j = top_left_i - 5, top_left_j - 5
        elif image[top_left_i - 5, top_left_j] == 255:
            top_left_i, top_left_j = top_left_i - 5, top_left_j
        elif image[top_left_i, top_left_j - 5] == 255:
            top_left_i, top_left_j = top_left_i, top_left_j - 5

        elif image[top_left_i - 3, top_left_j - 3] == 255:
            top_left_i, top_left_j = top_left_i - 3, top_left_j - 3
        elif image[top_left_i - 3, top_left_j] == 255:
            top_left_i, top_left_j = top_left_i - 3, top_left_j
        elif image[top_left_i, top_left_j - 3] == 255:
            top_left_i, top_left_j = top_left_i, top_left_j - 3
        else:
            break

    # Top Right Pixel
    top_right_i, top_right_j = center_i, center_j
    while True:
        if image[top_right_i - 10, top_right_j + 10] == 255:
            top_right_i, top_right_j = top_right_i - 10, top_right_j + 10
        elif image[top_right_i - 10, top_right_j] == 255:
            top_right_i, top_right_j = top_right_i - 10, top_right_j
        elif image[top_right_i, top_right_j + 10] == 255:
            top_right_i, top_right_j = top_right_i, top_right_j + 10

        elif image[top_right_i - 5, top_right_j + 5] == 255:
            top_right_i, top_right_j = top_right_i - 5, top_right_j + 5
        elif image[top_right_i - 5, top_right_j] == 255:
            top_right_i, top_right_j = top_right_i - 5, top_right_j
        elif image[top_right_i, top_right_j + 5] == 255:
            top_right_i, top_right_j = top_right_i, top_right_j + 5

        elif image[top_right_i - 3, top_right_j + 3] == 255:
            top_right_i, top_right_j = top_right_i - 3, top_right_j + 3
        elif image[top_right_i - 3, top_right_j] == 255:
            top_right_i, top_right_j = top_right_i - 3, top_right_j
        elif image[top_right_i, top_right_j + 3] == 255:
            top_right_i, top_right_j = top_right_i, top_right_j + 3
        else:
            break

    # Bottom Left Pixel
    bottom_left_i, bottom_left_j = center_i, center_j
    while True:
        if image[bottom_left_i + 10, bottom_left_j - 10] == 255:
            bottom_left_i, bottom_left_j = bottom_left_i + 10, bottom_left_j - 10
        elif image[bottom_left_i + 10, bottom_left_j] == 255:
            bottom_left_i, bottom_left_j = bottom_left_i + 10, bottom_left_j
        elif image[bottom_left_i, bottom_left_j - 10] == 255:
            bottom_left_i, bottom_left_j = bottom_left_i, bottom_left_j - 10

        elif image[bottom_left_i + 5, bottom_left_j - 5] == 255:
            bottom_left_i, bottom_left_j = bottom_left_i + 5, bottom_left_j - 5
        elif image[bottom_left_i + 5, bottom_left_j] == 255:
            bottom_left_i, bottom_left_j = bottom_left_i + 5, bottom_left_j
        elif image[bottom_left_i, bottom_left_j - 5] == 255:
            bottom_left_i, bottom_left_j = bottom_left_i, bottom_left_j - 5

        elif image[bottom_left_i + 3, bottom_left_j - 3] == 255:
            bottom_left_i, bottom_left_j = bottom_left_i + 3, bottom_left_j - 3
        elif image[bottom_left_i + 3, bottom_left_j] == 255:
            bottom_left_i, bottom_left_j = bottom_left_i + 3, bottom_left_j
        elif image[bottom_left_i, bottom_left_j - 3] == 255:
            bottom_left_i, bottom_left_j = bottom_left_i, bottom_left_j - 3
        else:
            break

    # Bottom Right Pixel
    bottom_right_i, bottom_right_j = center_i, center_j
    while True:
        if image[bottom_right_i + 10, bottom_right_j + 10] == 255:
            bottom_right_i, bottom_right_j = bottom_right_i + 10, bottom_right_j + 10
        elif image[bottom_right_i + 10, bottom_right_j] == 255:
            bottom_right_i, bottom_right_j = bottom_right_i + 10, bottom_right_j
        elif image[bottom_right_i, bottom_right_j + 10] == 255:
            bottom_right_i, bottom_right_j = bottom_right_i, bottom_right_j + 10

        elif image[bottom_right_i + 5, bottom_right_j + 5] == 255:
            bottom_right_i, bottom_right_j = bottom_right_i + 5, bottom_right_j + 5
        elif image[bottom_right_i + 5, bottom_right_j] == 255:
            bottom_right_i, bottom_right_j = bottom_right_i + 5, bottom_right_j
        elif image[bottom_right_i, bottom_right_j + 5] == 255:
            bottom_right_i, bottom_right_j = bottom_right_i, bottom_right_j + 5

        elif image[bottom_right_i + 3, bottom_right_j + 3] == 255:
            bottom_right_i, bottom_right_j = bottom_right_i + 3, bottom_right_j + 3
        elif image[bottom_right_i + 3, bottom_right_j] == 255:
            bottom_right_i, bottom_right_j = bottom_right_i + 3, bottom_right_j
        elif image[bottom_right_i, bottom_right_j + 3] == 255:
            bottom_right_i, bottom_right_j = bottom_right_i, bottom_right_j + 3
        else:
            break

    row_1 = np.min((top_left_i, top_right_i))  # Top row
    row_2 = np.max((bottom_left_i, bottom_right_j))  # Bottom row
    col_1 = np.min((top_left_j, bottom_left_j))  # Left column
    col_2 = np.max((top_right_j, bottom_right_j))  # Right column

    return [row_1, row_2, col_1, col_2]


def apply_adaptive_threshold(image):
    """
    Applies Adaptive Threshold with 7 x 7 kernel
    :param image:
    :return:
    """
    dimension = image.shape
    adaptive_threshold_image = np.copy(image)

    for i in range(2, dimension[0] - 2):
        for j in range(2, dimension[1] - 2):
            if adaptive_threshold_image[i][j] >= np.median(image[i-2:i+3, j-2:j+3]):
                adaptive_threshold_image[i][j] = 25
            else:
                adaptive_threshold_image[i][j] = 230
    return adaptive_threshold_image


def apply_bitwise_not(image):
    dimension = image.shape
    for i in range(3, dimension[0] - 3):
        for j in range(3, dimension[1] - 3):
            if image[i][j] == 25:
                image[i][j] = 230
            else:
                image[i][j] = 25
    return image


def apply_median_blur(image):
    dimension = image.shape
    median_blur_image = np.copy(image)
    for i in range(2, dimension[0] - 2):
        for j in range(2, dimension[1] - 2):
            median_blur_image[i][j] = np.median(image[i-2:i+3, j-2:j+3])
    return median_blur_image


def apply_mean_blur(image):
    dimension = image.shape
    mean_blur_image = np.copy(image)
    for i in range(2, dimension[0] - 2):
        for j in range(2, dimension[1] - 2):
            mean_blur_image[i][j] = np.mean(image[i-2:i+3, j-2:j+3])
    return mean_blur_image