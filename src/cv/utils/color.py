import colorsys

import numpy as np

np.random.seed(0)

def generate_bright_color():
    """
    Generate a random bright color in RGB format.

    Returns:
        Tuple of three integers representing bright RGB values.
    """
    # Generate a random hue in [0, 1) to cover the entire color spectrum
    hue = np.random.rand()

    # Set saturation and lightness to high values for brightness
    saturation = 0.8
    lightness = 0.7

    # Convert HSL to RGB
    rgb_color = colorsys.hls_to_rgb(hue, lightness, saturation)

    # Scale the values to the range [0, 255] and round to integers
    return tuple(int(val * 255) for val in rgb_color)