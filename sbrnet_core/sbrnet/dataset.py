import os
from functools import cache
from typing import Tuple

import numpy as np
import torch
import zarr
from tifffile import imread, TiffFile
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, folder):
        super(CustomDataset, self).__init__()
        self.directory = folder

    def __len__(self):
        data_dir = os.path.join(
            self.directory, "rfvbg"
        )  # bg refers to with background, rfv refers to refocused volume
        return len(
            [
                name
                for name in os.listdir(data_dir)
                if os.path.isfile(os.path.join(data_dir, name))
            ]
        )

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """General getitem function for sbrnet-related datasets. require the stack of lightfield views,
          and the refocused volume as inputs. and the the ground truth target.

        Args:
            index (int): index of the data. the input measurement data is stored in the format of meas_{index}.tiff,
            and the output is stored in the format of gt_vol_{index}.tiff. yours may change, so adjust this function accordingly.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: your data in torch tensor form normalized to [0,1] with 32bit float.
        """
        stack = (
            imread(os.path.join(self.directory, f"stackbg/meas_{index}.tiff")).astype(
                np.float32
            )
            / 255
        )
        stack = torch.from_numpy(stack)

        rfv = (
            imread(os.path.join(self.directory, f"rfvbg/meas_{index}.tiff")).astype(
                np.float32
            )
            / 255
        )

        rfv = torch.from_numpy(rfv)

        gt = (
            imread(os.path.join(self.directory, f"gt/gt_vol_{index}.tiff")).astype(
                np.float32
            )
            / 255
        )
        gt = torch.from_numpy(gt)

        return stack, rfv, gt


class PatchDataset(Dataset):
    def __init__(self, dataset: Dataset, patch_size: int):
        """Dataset class to include Poisson-Gaussian noise.

        Args:
            dataset (Dataset): the train split dataset after torch.utils.data.randomsplit for valid and train
        """
        self.dataset = dataset
        self.patch_size = patch_size

    @cache
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """retrieves a random patch of the data with size patch_size

        Args:
            idx (int): index of the data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: patch of the data with size patch_size.

            Note: One may realize that for RFV, we may crop out some peripheral information that's correlated
            with the GT, but we neglect this correlation as the axial shearing from the off-axis microlenses
            is not significant.
        """
        # Recipe for fast dataloading with zarr courtesy of Mitchell Gilmore mgilm0re@bu.edu
        stack_path = os.path.join(self.directory, f"stackbg/meas_{idx}.tiff")
        with TiffFile(stack_path) as img:
            stack = zarr.open(img.aszarr())

        rfv_path = os.path.join(self.directory, f"rfvbg/meas_{idx}.tiff")
        with TiffFile(rfv_path) as img:
            rfv = zarr.open(img.aszarr())

        gt_path = os.path.join(self.directory, f"gt/gt_vol_{idx}.tiff")
        with TiffFile(gt_path) as img:
            gt = zarr.open(img.aszarr())

        # uniformly sample a 224 patch
        row_start = torch.randint(0, stack.shape[-2] - self.patch_size, (1,))
        col_start = torch.randint(0, stack.shape[-1] - self.patch_size, (1,))

        row_slice = slice(row_start, row_start + self.patch_size)
        col_slice = slice(col_start, col_start + self.patch_size)

        stack = torch.from_numpy(
            stack[:, row_slice, col_slice].astype(np.float32) / 255
        )
        rfv = torch.from_numpy(rfv[:, row_slice, col_slice].astype(np.float32) / 255)
        gt = torch.from_numpy(gt[:, row_slice, col_slice].astype(np.float32) / 255)

        return stack, rfv, gt

    def __len__(self):
        return len(self.dataset)
