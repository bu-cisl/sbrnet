import os
import torch
import numpy as np
from torch.utils.data import Dataset

from tifffile import imread


# calibrated parameters for poisson gaussian noise model
# cite
A_STD = 5.7092e-5
A_MEAN = 1.49e-4
B_STD = 2.7754e-6
B_MEAN = 5.41e-6


class CustomDataset(Dataset):
    def __init__(self, folder):
        super(CustomDataset, self).__init__()
        self.directory = folder

    def __len__(self):
        data_dir = os.path.join(self.directory, "rfvbg")  # bg refers to with background
        return len(
            [
                name
                for name in os.listdir(data_dir)
                if os.path.isfile(os.path.join(data_dir, name))
            ]
        )

    def __getitem__(self, index):
        stack = imread(
            os.path.join(self.directory, f"stackbg/meas_{index}.tiff")
        ).astype(np.float32)
        stack = (stack - stack.min()) / (stack.max() - stack.min())
        stack = torch.from_numpy(stack)

        rfv = imread(os.path.join(self.directory, f"rfvbg/meas_{index}.tiff")).astype(
            np.float32
        )

        rfv = (rfv - rfv.min()) / (rfv.max() - rfv.min())
        rfv = torch.from_numpy(rfv)

        gt = imread(os.path.join(self.directory, f"gt/gt_vol_{index}.tiff")).astype(
            np.float32
        )
        gt = (gt - gt.min()) / (gt.max() - gt.min())
        gt = torch.from_numpy(gt)

        return stack, rfv, gt


class MySubset(Dataset):
    def __init__(self, dataset: Dataset, is_val: bool):
        """Dataset class to include Poisson-Gaussian noise.

        Args:
            dataset (Dataset): the complete clean dataset
            is_val (boolean): if true, the dataset is for validation, else for training.
                              for validation, do not do any cropping
        """
        self.dataset = dataset
        self.is_val = is_val

    def __getitem__(self, idx):
        patch_size = 224

        aa = torch.randn(1) * A_STD + A_MEAN
        bb = torch.randn(1) * B_STD + B_MEAN

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
            a = torch.randint(0, dim[1] - patch_size, (1,))
            b = torch.randint(0, dim[2] - patch_size, (1,))

            stack = stack[:, a : a + patch_size, b : b + patch_size]
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
