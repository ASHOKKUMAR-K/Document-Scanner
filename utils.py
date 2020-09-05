def dimension_of_matrix(image):
    return image.shape


def choose_optimal_dimension(original_dimension, to_height, to_width):
    height = original_dimension[0]
    width = original_dimension[1]
    row_pixels = 1
    col_pixels = 1

    if height > to_height:
        row_pixels = height // to_height
    if width > to_width:
        col_pixels = width // to_width

    optimal_dimension = (height // row_pixels, width // col_pixels)

    return row_pixels, col_pixels, optimal_dimension