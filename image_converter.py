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
            # print(end_i, end_j)

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
    dimension = utils.dimension_of_matrix(gray_image)
    blurred_image = np.copy(gray_image)

    for i in range(1, dimension[0] - 1):
        for j in range(1, dimension[1] - 1):
            blurred_image[i][j] = (np.sum(gray_image[i - 1:i + 2, j - 1:j + 2]) // 9) // 1
    return np.array(blurred_image, np.uint8)


def apply_gaussian_blur_effects_to(gray_image):
    """
    Smoothens the image with Gaussian Blur
    :param gray_image: Gray Scale Image Matrix, type: numpy.ndarray
    :return: Smoothened Image Matrix
    """
    dimension = utils.dimension_of_matrix(gray_image)
    blurred_image = np.copy(gray_image)
    print("Gaussian")

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