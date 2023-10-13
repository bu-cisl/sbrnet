import os
import numpy as np
from typing import Tuple

from tifffile import imread, imwrite
import imageio.v3 as iio
from scipy.signal import fftconvolve
from multiprocessing import Pool


def read_tiff(path: str) -> np.ndarray:
    """read a .tiff image into a numpy array

    Args:
        path (str): path to the image

    Returns:
        np.ndarray: numpy tiff image
    """
    return imread(path)


def read_png(path: str) -> np.ndarray:
    """read a .tiff image into a numpy array

    Args:
        path (str): path to the image

    Returns:
        np.ndarray: numpy tiff image
    """

    return iio.imread(path)


def uint8_to_float(x: np.ndarray) -> np.ndarray:
    """

    Args:
        x (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    x_float = x.astype(np.float32)
    x_float /= 0xFF
    return x_float


def full_read(data_path: str) -> np.ndarray:
    """combines reading and normalizing

    Args:
        path (str): _description_

    Returns:
        np.ndarray: _description_
    """
    if data_path.lower().endswith((".tiff", ".tif")):
        return uint8_to_float(read_tiff(data_path))
    if data_path.lower().endswith(".png"):
        return uint8_to_float(read_png(data_path))


def write_tiff(x: np.ndarray, path: str) -> None:
    """write a numpy array to a tiff file

    Args:
        x (np.ndarray): the numpy array you want to save
    """
    x = (255 * (linear_normalize(x))).astype("uint8")
    imwrite(path, x)


def linear_normalize(x: np.ndarray) -> np.ndarray:
    """linearly normalizes the array

    Args:
        x (np.ndarray): any sized array

    Returns:
        np.ndarray: same sized array
    """
    return (x - x.min()) / (x.max() - x.min() + np.finfo(float).eps)


def clip(x: np.ndarray, low: float, high: float) -> np.ndarray:
    """Clips the array

    Args:
        x (np.ndarray): Array to be clipped
        low (float): Below this is low
        high (float): Above this is high

    Returns:
        np.ndarray: Clipped array
    """
    return np.minimum(np.maximum(x, low), high)


def crop(arr: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    """Crops the center of the array

    Args:
        arr (np.ndarray): _description_
        new_height (int): _description_
        new_width (int): _description_

    Returns:
        np.ndarray: _description_
    """
    height, width = arr.shape

    start_row = (height - new_height) // 2
    start_col = (width - new_width) // 2

    cropped_arr = arr[
        start_row : start_row + new_height, start_col : start_col + new_width
    ]

    return cropped_arr


def pad_3d_array_to_size(
    arr: np.ndarray, target_shape: Tuple[int, int, int]
) -> np.ndarray:
    """
    Pad a 3D NumPy array `arr` to the desired shape `target_shape` along the last two dimensions with zeros.

    Parameters:
    - arr: 3D NumPy array to be padded.
    - target_shape: Tuple (d, H, W) specifying the desired shape.

    Returns:
    - Padded 3D NumPy array of shape (d, H, W).
    """
    d, r, c = arr.shape
    d_target, H, W = target_shape

    # Calculate the amount of padding needed for each 2D slice (r, c) independently
    pad_height = max(0, H - r)
    pad_width = max(0, W - c)

    # Calculate the padding for each side (top, bottom, left, right)
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # Initialize a new array with the target shape, filled with zeros
    padded_array = np.zeros((d_target, H, W), dtype=arr.dtype)

    # Iterate through the 2D slices along the first dimension (d) and pad each slice
    for i in range(d):
        padded_array[i] = np.pad(
            arr[i],
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )

    return padded_array


def pad0(arr: np.ndarray) -> np.ndarray:
    """Pad a 2d numpy array for fft2d

    Args:
        arr (np.ndarray): Array to be padded
        pad_height (int): _description_
        pad_width (int): _description_
        value (int, optional): _description_. Defaults to 0.

    Returns:
        np.ndarray: _description_
    """
    pad_height = arr.shape[0] // 2
    pad_width = arr.shape[1] // 2
    pad_tuple = ((pad_height, pad_height), (pad_width, pad_width))
    padded_arr = np.pad(
        arr,
        pad_tuple,
        mode="constant",
        constant_values=0,
    )

    return padded_arr


def fft2d(x: np.ndarray) -> np.ndarray:
    """2D Fast Fourier Transform

    Args:
        x (np.ndarray): _description_

    Returns:
        np.ndarray: complex-valued 2D FFT
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))


def ifft2d(x: np.ndarray) -> np.ndarray:
    """2D Inverse Fast Fourier Transform

    Args:
        x (np.ndarray): _description_

    Returns:
        np.ndarray: complex-valued 2D iFFT
    """
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))


def power_normalize(x: np.ndarray) -> np.ndarray:
    """divides the array by its sum.

    Args:
        x (np.ndarray): any real valued numpy array at all

    Returns:
        np.ndarray: all elements sum to 1
    """
    return x / np.sum(x)


def normalize_psf_power(psf: np.ndarray) -> np.ndarray:
    """Normalizes each slice of the PSF by the power

    Args:
        psf (np.ndarray): 3D stack of PSFs or just a single 2D PSF

    Returns:
        np.ndarray: Power-normalized PSF.
    """
    if psf.ndim == 2:
        return power_normalize(psf)

    for z in range(psf.shape[0]):
        tmp = psf[z, :, :]
        psf[z, :, :] = power_normalize(tmp)
    return psf


def conv2d(obj: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D convolution between an "object" and a kernel

    Args:
        obj (np.ndarray): _description_
        kernel (np.ndarray): _description_

    Returns:
        np.ndarray: The convolution between the obj and the kernel
    """
    assert obj.shape == kernel.shape
    return fftconvolve(obj, kernel, mode="same")  # faster than own implementation
    # return crop(np.real(ifft2d(fft2d(pad0(obj)) * fft2d(pad0(kernel)))), *obj.shape)


def lsi_fwd_mdl(obj: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Linear shift-invariant (LSI) forward model to propagate an object to the image plane via a PSF

    Args:
        obj (np.ndarray): object. can be 2D or 3D (z,x,y)
        psf (np.ndarray): point spread function (PSF) has to be the same shape as the object.

    Returns:
        np.ndarray: LSI measurement
    """

    meas = np.zeros((obj.shape[1:]))
    for z in range(obj.shape[0]):
        meas += conv2d(obj[z, :, :], psf[z, :, :])
    return meas / np.max(meas)


def process_slice(z, obj, psf):
    return conv2d(obj[z, :, :], psf[z, :, :])


@DeprecationWarning
def lsi_fwd_mdl_multiproc(obj: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Linear shift-invariant (LSI) forward model to propagate an object to the image plane via a PSF

    Args:
        obj (np.ndarray): object. can be 2D or 3D (z, x, y)
        psf (np.ndarray): point spread function (PSF) has to be the same shape as the object.

    Returns:
        np.ndarray: LSI measurement
    """

    with Pool(processes=2) as pool:  # Adjust the number of processes as needed
        results = pool.starmap(
            process_slice, [(z, obj, psf) for z in range(obj.shape[0])]
        )

    meas = np.sum(results, axis=0)
    return meas / np.max(meas)


def shift_array(arr, shift_x, shift_y):
    """
    Shift a NumPy array without circular wrapping.

    Args:
        arr (numpy.ndarray): The input array.
        shift_x (int): The horizontal shift (positive values shift to the right, negative to the left).
        shift_y (int): The vertical shift (positive values shift downward, negative upward).

    Returns:
        numpy.ndarray: The shifted array.
    """
    if shift_x == 0 and shift_y == 0:
        return arr  # No shift needed

    h, w = arr.shape
    shifted_arr = np.zeros_like(arr)

    # Calculate slices for rows and columns
    if shift_x >= 0:
        x_start_src, x_end_src, x_start_dst, x_end_dst = 0, w - shift_x, shift_x, w
    else:
        x_start_src, x_end_src, x_start_dst, x_end_dst = -shift_x, w, 0, w + shift_x

    if shift_y >= 0:
        y_start_src, y_end_src, y_start_dst, y_end_dst = 0, h - shift_y, shift_y, h
    else:
        y_start_src, y_end_src, y_start_dst, y_end_dst = -shift_y, h, 0, h + shift_y

    # Copy the shifted region from the source to the destination
    shifted_arr[y_start_dst:y_end_dst, x_start_dst:x_end_dst] = arr[
        y_start_src:y_end_src, x_start_src:x_end_src
    ]

    return shifted_arr
