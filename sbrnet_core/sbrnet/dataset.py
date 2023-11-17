# Importing modules
import os
from functools import cached_property, cache
from typing import Tuple
from pandas import DataFrame, read_parquet
import numpy as np
import torch
import zarr
from tifffile import imread, TiffFile
from torch.utils.data import Dataset


# Creating CustomDataset Class
class CustomDataset(Dataset):
    """
    Custom dataset class for handling image datasets consisting of a stack of lightfield views, a refocused volume, 
    and a ground truth target. Class reads data from file paths specified in a DataFrame.
    """
    def __init__(self, df_path: str):
        """
        Initialize the CustomDataset object.

        Args:
            df_path (str): Path to a DataFrame in parquet format containing file paths to image data.
        """
        super(CustomDataset, self).__init__()
        self.df = read_parquet(df_path)  # Read the DataFrame from the given path

    def __len__(self):
        """ Return the number of items in the dataset. """
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """General getitem function for sbrnet-related datasets. require the stack of lightfield views,
          and the refocused volume as inputs. and the the ground truth target.

        Args:
            index (int): Index of the data item to retrieve. Note input measurement data is stored in format 
            of meas_{index}.tiff and output is stored in the format of gt_vol_{index}.tiff

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing stack, rfv, and gt data
            all as normalized 32-bit float torch tensors to the [0, 1] range.
        """

        # Load and normalize the stack images from file path
        stack = imread(self.df["stack_path"].iloc[index]).astype(np.float32) / 255
        stack = torch.from_numpy(stack)

        # Load and normalize the rfv image from file path
        rfv = imread(self.df["rfv_path"].iloc[index]).astype(np.float32) / 255
        rfv = torch.from_numpy(rfv)

        # Load and normalize gt images from file path
        gt = imread(self.df["gt_path"].iloc[index]).astype(np.float32) / 255
        gt = torch.from_numpy(gt)

        return stack, rfv, gt


# Creating ZarrData Class
class ZarrData:
    """
    A class for efficient data access and loading using Zarr designed to work with image data stored in TIFF format.
    """
    def __init__(self, df: DataFrame, datatype: str):
        """
        Initialize the ZarrData object.

        Args:
            df (DataFrame): A pandas DataFrame containing file paths to image data.
            datatype (str): Specifies the type of data to retrieve. Must be one of 'stack', 'rfv', 'gt'.
        """
        self.df = df
        if datatype not in ["stack", "rfv", "gt"]:
            raise ValueError("datatype must be one of stack, rfv, gt")
        self.datatype = datatype
        self.open_zarrs = []

    # NOTE: ensure cache is larger than number of items
    @cache
    def __getitem__(self, index: int):
        """
        Retrieve the dataset item at the specified index using Zarr for efficient data loading.

        Args:
            index (int): Index of the data item to retrieve.

        Returns:
            The data item loaded as a Zarr array.
        """
        path = self.df[self.datatype + "_path"].iloc[index]
        with TiffFile(path) as img:
            return zarr.open(img.aszarr())
        
        
# Creating PatchDataset Class
class PatchDataset(Dataset):
    """
    A dataset class for handling patch-based data, useful for training on cropped portions of images.
    """
    def __init__(self, dataset: Dataset, df_path: str, patch_size: int):
        """
        Initialize the PatchDataset object for handling patch data (cropping).

        Args:
            dataset (Dataset): The dataset to use, typically a train split after applying a random split.
            df_path (str): Path to a DataFrame in parquet format.
            patch_size (int): Size of the patches to be cropped from the dataset.
        """
        self.dataset = dataset
        self.df = read_parquet(df_path)
        self.patch_size = patch_size

    @cached_property
    def stack(self) -> ZarrData:
        """Cached property to access stack data as ZarrData."""
        return ZarrData(self.df, "stack")

    @cached_property
    def rfv(self) -> ZarrData:
        """ Cached property to access refocused volume data as ZarrData. """
        return ZarrData(self.df, "rfv")

    @cached_property
    def gt(self) -> ZarrData:
        """Cached property to access ground truth data as ZarrData. """
        return ZarrData(self.df, "gt")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves a random patch of the data with size patch_size.

        Args:
            idx (int): Index of the data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing a patch of the stack data,
            refocused volume, and ground truth, all as torch tensors.
        """
        # # Retrieve the data as Zarr arrays and sample a patch from data
        # Courtesy of Mitchell Gilmore mgilm0re@bu.edu
        stack = self.stack[idx]
        rfv = self.rfv[idx]
        gt = self.gt[idx]

        # Uniformly sample a patch
        # Randomly select starting row and col index for patch within the bounds of the image
        row_start = torch.randint(0, stack.shape[-2] - self.patch_size, (1,))
        col_start = torch.randint(0, stack.shape[-1] - self.patch_size, (1,))

        # Create a slice object for rows and columns, defining the range of the patch.
        row_slice = slice(row_start, row_start + self.patch_size)
        col_slice = slice(col_start, col_start + self.patch_size)

        # Extract patch from stack, rfv, gt and normalize its values to the range [0, 1].
        stack = torch.from_numpy(stack[:, row_slice, col_slice].astype(np.float32) / 255)
        rfv = torch.from_numpy(rfv[:, row_slice, col_slice].astype(np.float32) / 255)
        gt = torch.from_numpy(gt[:, row_slice, col_slice].astype(np.float32) / 255)

        return stack, rfv, gt

    def __len__(self):
        """ Return the number of items in the dataset. """
        return len(self.dataset)
