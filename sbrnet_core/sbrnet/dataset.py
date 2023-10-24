import os
import torch
import numpy as np
from torch.utils.data import Dataset
from skimage import io

from sbrnet_core.utils.utils import full_read_tiff

# calibrated parameters for poisson gaussian noise model
# cite
A_STD = 5.7092e-5
A_MEAN = 1.49e-4
B_STD = 2.7754e-6
B_MEAN = 5.41e-6

print("here")


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
        stack = io.imread(os.path.join(self.directory, f"stackbg/meas_{index}.tiff"))
        stack = (stack - stack.min()) / (stack.max() - stack.min()).astype(np.int16)
        stack = torch.from_numpy(stack)

        rfv = io.imread(os.path.join(self.directory, f"rfvbg/meas_{index}.tiff"))
        rfv = (rfv - rfv.min()) / (rfv.max() - rfv.min()).astype(np.int16)
        rfv = torch.from_numpy(rfv)

        gt = io.imread(os.path.join(self.directory, f"gt/gt_vol_{index}.tiff"))
        gt = (gt - gt.min()) / (gt.max() - gt.min()).astype(np.int16)
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
            stack, rfv, gt = self.dataset.__getitem__(idx)
            stack += torch.sqrt(aa * stack + bb) * torch.randn(stack.shape)
            rfv += torch.sqrt(aa * rfv + bb) * torch.randn(rfv.shape) / 3
            return stack, rfv, gt
        else:
            stack, rfv, gt = self.dataset.__getitem__(idx)
            dim = stack.shape
            a = torch.randint(0, dim[1] - patch_size, (1,))
            b = torch.randint(0, dim[2] - patch_size, (1,))

            stack = stack[:, a : a + patch_size, b : b + patch_size]
            stack += torch.sqrt(aa * stack + bb) * torch.randn(stack.shape)
            rfv = rfv[:, a : a + patch_size, b : b + patch_size]
            rfv += torch.sqrt(aa * rfv + bb) * torch.randn(rfv.shape) / 3
            return stack, rfv, gt[:, a : a + patch_size, b : b + patch_size]

    def __len__(self):
        return len(self.dataset)
