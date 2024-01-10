import numpy as np
from typing import Tuple, List
from skimage.feature import peak_local_max
from sbrnet_core.utils.constants import CM2_SIZE, FOCUS_LOC, NUM_VIEWS
from sbrnet_core.utils import (
    linear_normalize,
    read_tiff,
    uint8_to_float,
    normalize_psf_power,
    shift_array,
)


# Functions
def load_data(path: str) -> np.ndarray:
    """reads and normalizes data

    Args:
        path (str): path to data

    Returns:
        np.ndarray: numpy.ndarray
    """
    return uint8_to_float(read_tiff(path))


def load_psf(path: str) -> np.ndarray:
    """reads and prepares PSF stack

    Args:
        path (str): path to psf file

    Returns:
        np.ndarray: psf stack
    """
    return normalize_psf_power(linear_normalize(read_tiff(path)))


def crop_views(im: np.ndarray, crop_size: int = 512) -> np.ndarray:
    # Your code here
    """takes a 2076x 3088 CM2v2 image and crops it from left to right starting at the top

    Args:
        im (np.ndarray): the 2076x3088 image
        crop_size (int): side length of the square crop
    Returns:
        np.ndarray: [NUM_VIEWS, crop_size, crop_size]
    """
    stack = np.zeros((NUM_VIEWS, crop_size, crop_size))
    for i, point in enumerate(FOCUS_LOC):
        x, y = point
        x_min = x - crop_size // 2
        x_max = x + crop_size // 2
        y_min = y - crop_size // 2
        y_max = y + crop_size // 2

        # Crop the region from the larger array and store it in the 3D numpy array
        stack[i, :, :] = im[x_min:x_max, y_min:y_max]
    return stack


def get_coord_max(im: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        im (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    fixed_meas = im.copy()
    fixed_meas[im <= 0.25] = 0
    return peak_local_max(fixed_meas, min_distance=5)


def get_meas_mean(im: np.ndarray) -> np.float32:
    """looks at the peaks of the signal in the freespace measurement and gets the average
    This is for computing the signal to background ratio (SBR).
    Args:
        im (np.ndarray): 2076x3088 CM2v2 image

    Returns:
        np.float32: average of all the peaks of the signal
    """
    # remove noisy background that may contribute to the peak finding
    # im[im < 0.25] = 0
    coords = get_coord_max(im)

    # Step 1: Extract pixel values for the coordinates
    pixel_values = [im[x, y] for x, y in coords]

    # Step 2: Compute the average of the pixel values
    return np.mean(pixel_values)


def get_background_mean(im: np.ndarray) -> np.float32:
    """takes the value noise image and gets its mean value

    Args:
        im (np.ndarray): value noise sample, usually 600x600

    Returns:
        np.float32: the mean of all the values
    """
    return np.mean(im)


def make_bg_img(
    value_img: np.ndarray,
    lens_ap: np.ndarray,
    mla_ap: np.ndarray,
    crop_size: int = 600,
) -> Tuple[np.ndarray, np.float32]:
    """make 9x9 background only cm2 image. use an apodization mask for the lens FOV

    Args:
        value_img (np.ndarray): 600x600 synthetic background
        lens_ap (np.ndarray): gaussian mask for

    Returns:
        np.ndarray: 2076x3088 cm2 background measurement
    """
    assert value_img.shape == lens_ap.shape
    bg_mask = np.zeros((CM2_SIZE))
    for point in FOCUS_LOC:
        x, y = point
        x_min = x - crop_size // 2
        x_max = x + crop_size // 2
        y_min = y - crop_size // 2
        y_max = y + crop_size // 2

        bg_mask[x_min:x_max, y_min:y_max] = value_img * lens_ap

    bg_mean = get_background_mean(value_img)
    return linear_normalize(bg_mask * mla_ap), bg_mean


def make_measurement(
    freespace_img: np.ndarray,
    bg_img: np.ndarray,
    SBR: np.float32,
    bg_mean: np.float32,
) -> np.ndarray:
    """takes free space measurement and background measurement and makes scattering measurement

    Args:
        freespace_img (np.ndarray): convolution between freespace PSF and GT volume
        bg_img (np.ndarray): value noise with apodization
        SBR (np.float32): a float for the signal to background ratio
        bg_mean (np.float32): float for the mean values in the raw value noise image

    Returns:
        np.ndarray: scattering measurement simulation with value noise modeling the low-contrast effect
        of scattering. same shape as freespace_img.
    """
    meas_mean = get_meas_mean(freespace_img)
    S = (bg_mean * SBR - bg_mean) / meas_mean
    scattering_meas = linear_normalize(S * freespace_img + bg_img)
    return scattering_meas


def lf_refocus_step(lf: np.ndarray, shift: int) -> np.ndarray:
    """One step of the shift-and-add operation for light field refocusing

    Args:
        lf (np.ndarray): 4D array, [row_mla, col_mla, H, W]
        shift (int): number of pixels to shift before adding

    Returns:
        np.ndarray: Shift and added result [H,W]. This is one slice of the refocused volume
    """
    c_lf = lf.shape[0] // 2 + 1  # assume row_mla == col_mla

    out = np.zeros((lf.shape[-2:]))
    for r in range(lf.shape[0]):
        for c in range(lf.shape[1]):
            shift_tuple = (-1 * (c_lf - (c + 1)) * shift, -1 * (c_lf - (r + 1)) * shift)
            out += shift_array(lf[r, c, :, :], *shift_tuple)
    return out


def lf_refocus_volume(
    lf: np.ndarray, z_slices: int, max_shift: int, mla_size: tuple = (3, 3)
) -> np.ndarray:
    """Light-field (LF) refocusing. assuming a square grid of microlenses. Implements the "shift-and-add" algorithm
    that is similar to inverse radon transform of "smearing" the tomograms along the captured angle.

    Args:
        lf (np.ndarray): Cropped light field [num_views, H, W]
        max_shift (int): Usually the ceil of half the number of z-slices in the object volume

    Returns:
        np.ndarray: Refocused volume (RFV) [z_slices, H, W]
    """
    lf = lf.reshape((*mla_size, *lf.shape[-2:]))

    rfv = np.zeros((z_slices, *lf.shape[-2:]))

    for ii, z in enumerate(range(z_slices)):
        rfv[z, :, :] = lf_refocus_step(lf=lf, shift=ii - max_shift + 1)

    return linear_normalize(rfv)


def zero_slices_not_in_list(arr, indices_to_keep):
    """
    Zero out slices in a 3D array along the first axis (c-axis) that are not in the specified list of indices.

    Parameters:
    - arr (numpy.ndarray): The 3D array with shape (c, h, w) where c is the number of slices along the c-axis,
                          and h, w are the height and width of each slice.
    - indices_to_keep (list): A list of integers representing the indices along the c-axis that should be retained.
                              Slices corresponding to indices not in this list will be set to zero.

    Returns:
    numpy.ndarray: A modified array where slices not indexed in the provided list are set to zero.

    Example:
    >>> c, h, w = 3, 4, 5
    >>> array = np.random.rand(c, h, w)
    >>> indices_to_keep = [0, 2]
    >>> result_array = zero_slices_not_in_list(array, indices_to_keep)
    >>> print(result_array)
    """
    # Create a boolean mask where True corresponds to indices to keep
    mask = np.isin(np.arange(arr.shape[0]), indices_to_keep)

    # Use the mask to zero out slices not in the list
    arr[~mask, :, :] = 0

    return arr

def attenuate_psf(PSF: np.ndarray, scattering_length: float, z_sampling: float) -> np.ndarray:
    """_summary_

    Args:
        PSF (np.ndarray): z, x,y
        scattering_length (float): in microns.

    Returns:
        np.ndarray: attenuated PSF
    """
    PSF_atten = PSF.copy()
    for z in range(PSF.shape[0]):
        PSF_atten[z, :, :] *= np.exp(-z*z_sampling / scattering_length)
    return PSF_atten
