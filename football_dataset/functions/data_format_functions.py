import numpy as np


def scale_num(num, min_input, max_input, min_output, max_output):
    scaled_num = ((num - min_input) / (max_input - min_input)) * (
        max_output - min_output
    ) + min_output

    return scaled_num
