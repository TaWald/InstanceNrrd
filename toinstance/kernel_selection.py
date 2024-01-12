from enum import Enum
from typing import Callable
from skimage import morphology as morph
import numpy as np


kernel_choices = ["ball", "cube", "diamond", "octahedron", "none"]


def get_kernel_from_str(kernel: str) -> Callable[[int, np.dtype], np.ndarray] | None:
    """
    Takes a string and returns the corresponding Kernel ENUM.
    """
    if kernel == "ball":
        return morph.ball
    elif kernel == "cube":
        return morph.cube
    elif kernel == "diamond":
        return morph.diamond
    elif kernel == "octahedron":
        return morph.octahedron
    elif kernel == "none":
        return None


def get_np_arr_from_kernel(
    kernel: Callable[[int, np.dtype], np.ndarray] | None, radius: int = 3, dtype: np.dtype = np.uint8
):
    """
    Takes the Kernel ENUM containing functions and creates the actual numpy kernel.
    This numpy kernel will be used for the connected components or the dilation.
    """
    if kernel is None:
        return None
    return kernel(radius, dtype)
