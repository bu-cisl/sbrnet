import os
from functools import cached_property, cache
from typing import Tuple
from pandas import DataFrame, read_parquet

import numpy as np
import torch
import zarr
from tifffile import imread, TiffFile
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, config: dict):
        super(CustomDataset, self).__init__()
        self.df = read_parquet(config["dataset_pq"])
        self.scattering = config["scattering"]

    def __len__(self):
        return len(self.df)

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
            imread(self.df[f"stack_{self.scattering}_path"].iloc[index]).astype(
                np.float32
            )
            / 0xFFFF
        )
        stack = torch.from_numpy(stack)

        rfv = (
            imread(self.df[f"rfv_{self.scattering}_path"].iloc[index]).astype(
                np.float32
            )
            / 0xFFFF
        )
        rfv = torch.from_numpy(rfv)

        gt = imread(self.df["gt_path"].iloc[index]).astype(np.float32) / 0xFFFF
        gt = torch.from_numpy(gt)

        return stack, rfv, gt


class ZarrData:
    def __init__(self, df: DataFrame, datatype: str, pattern: str):
        self.df = df

        if datatype not in ["stack", "rfv", "gt"]:
            raise ValueError("datatype must be one of stack, rfv, gt")

        self.datatype = datatype
        self.pattern = pattern
        self.open_zarrs = []

    # NOTE: ensure cache is larger than number of items
    @cache
    def __getitem__(self, index: int):
        path = self.df[self.datatype + self.pattern].iloc[index]
        with TiffFile(path) as img:
            return zarr.open(img.aszarr())


class PatchDataset(Dataset):
    def __init__(self, dataset: Dataset, config: dict):
        """Dataset class for patch data (cropping).

        Args:
            dataset (Dataset): the train split dataset after torch.utils.data.randomsplit for valid and train
        """
        self.dataset = dataset
        self.df = read_parquet(config["dataset_pq"])
        self.patch_size = config["patch_size"]
        self.scattering = config[
            "scattering"
        ]  # whether the data is free space or scattering

    @cached_property
    def stack(self) -> ZarrData:
        return ZarrData(self.df, "stack", pattern=f"_{self.scattering}_path")

    @cached_property
    def rfv(self) -> ZarrData:
        return ZarrData(self.df, "rfv", pattern=f"_{self.scattering}_path")

    @cached_property
    def gt(self) -> ZarrData:
        return ZarrData(self.df, "gt", "_path")

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
        stack = self.stack[idx]
        rfv = self.rfv[idx]
        gt = self.gt[idx]

        # uniformly sample a patch
        row_start = torch.randint(0, stack.shape[-2] - self.patch_size, (1,))
        col_start = torch.randint(0, stack.shape[-1] - self.patch_size, (1,))

        row_slice = slice(row_start, row_start + self.patch_size)
        col_slice = slice(col_start, col_start + self.patch_size)

        stack = torch.from_numpy(
            stack[:, row_slice, col_slice].astype(np.float32) / 0xFFFF
        )
        rfv = torch.from_numpy(rfv[:, row_slice, col_slice].astype(np.float32) / 0xFFFF)
        gt = torch.from_numpy(gt[:, row_slice, col_slice].astype(np.float32) / 0xFFFF)

        return stack, rfv, gt

    def __len__(self):
        return len(self.dataset)