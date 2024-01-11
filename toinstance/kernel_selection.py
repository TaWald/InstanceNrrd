from enum import Enum
from skimage import morphology as morph
import numpy as np


class Kernel(Enum):
    ball = morph.ball
    cube = morph.cube
    diamond = morph.diamond
    octahedron = morph.octahedron
    none = None


def get_np_arr_from_kernel(kernel: Kernel | None, radius: int = 3, dtype: np.dtype = np.uint8):
    """
    Takes the Kernel ENUM containing functions and creates the actual numpy kernel.
    This numpy kernel will be used for the connected components or the dilation.
    """
    if kernel is None:
        return None
    return kernel(radius, dtype)
