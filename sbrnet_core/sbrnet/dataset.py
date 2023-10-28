from typing import Tuple
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tifffile import imread


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

    def __getitem__(self, idx):
        if self.is_val:
            stack, rfv, gt = self.dataset[idx]
            stack += torch.sqrt(torch.clamp(aa * stack + bb, min=0)) * torch.randn(
                stack.shape
            )
            rfv += (
                torch.sqrt(torch.clamp(aa * rfv + bb, min=0))
                * torch.randn(rfv.shape)
                / 3
            )
            return stack, rfv, gt
        else:
            stack, rfv, gt = self.dataset.__getitem__(idx)
            dim = stack.shape
            a = torch.randint(0, dim[1] - self.patch_size, (1,))
            b = torch.randint(0, dim[2] - self.patch_size, (1,))

            stack = stack[:, a : a + self.patch_size, b : b + patch_size]
            stack += torch.sqrt(torch.clamp(aa * stack + bb, min=0)) * torch.randn(
                stack.shape
            )
            rfv = rfv[:, a : a + patch_size, b : b + patch_size]
            rfv += (
                torch.sqrt(torch.clamp(aa * rfv + bb, min=0))
                * torch.randn(rfv.shape)
                / 3
            )
            return stack, rfv, gt[:, a : a + patch_size, b : b + patch_size]

    def __len__(self):
        return len(self.dataset)


class ValidationDataset(Dataset):
    def __init__(self, folder) -> None:
        super(ValidationDataset).__init__()
        self.directory = folder
        stack_folder = os.path.join(self.directory, "stackbg")
        rfv_folder = os.path.join(self.directory, "rfvbg")
        gt_folder = os.path.join(self.directory, "gt")

        # Get the number of files in each folder
        stack_files = os.listdir(stack_folder)
        rfv_files = os.listdir(rfv_folder)
        gt_files = os.listdir(gt_folder)

        # Ensure all folders have the same number of files
        assert (
            len(stack_files) == len(rfv_files) == len(gt_files)
        ), "Not all folders have the same number of files."

        # You can also print the counts if needed
        print(f"stackbg folder has {len(stack_files)} files.")
        print(f"rfvbg folder has {len(rfv_files)} files.")
        print(f"gt folder has {len(gt_files)} files")

    def __len__(self) -> int:
        data_dir = os.path.join(self.directory, "rfvbg")  # bg refers to with background
        return sum(1 for entry in os.scandir(data_dir) if entry.is_file())

    def __getitem__(self, index):
        stack = torch.from_numpy(
            imread(os.path.join(self.directory, f"stackbg/meas_{index}.tiff")).astype(
                np.float32
            )
            / 255
        )

        rfv = torch.from_numpy(
            imread(os.path.join(self.directory, f"rfvbg/meas_{index}.tiff")).astype(
                np.float32
            )
            / 255
        )

        gt = torch.from_numpy(
            imread(os.path.join(self.directory, f"gt/gt_vol_{index}.tiff")).astype(
                np.float32
            )
            / 255
        )

        return stack, rfv, gt
