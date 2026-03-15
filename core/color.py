import numpy as np

# Thresholds from Kopf-Lischinski 2011 paper
Y_THRESHOLD = 48 / 255
U_THRESHOLD = 7 / 255
V_THRESHOLD = 6 / 255

def rgb_to_yuv(rgb):
    """
    Converts RGB color to YUV space.
    Input rgb: tuple or numpy array (0-255)
    Returns: numpy array (0-1)
    """
    r, g, b = np.array(rgb) / 255.0
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b
    v = 0.615 * r - 0.51499 * g - 0.10001 * b
    return np.array([y, u, v])

def are_colors_similar(yuv1, yuv2):
    """
    Checks if two YUV colors are similar based on the paper's thresholds.
    """
    diff = np.abs(yuv1 - yuv2)
    return (diff[0] < Y_THRESHOLD and 
            diff[1] < U_THRESHOLD and 
            diff[2] < V_THRESHOLD)
