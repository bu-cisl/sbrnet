from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor, Module


class PoissonGaussianNoiseModel(Module):
    """Poisson-Gaussian noise model for CM2 sensor: https://ieeexplore.ieee.org/document/4623175"""

    def __init__(self, config: dict):
        super().__init__()

        self.a_std = config.get("A_STD")
        self.a_mean = config.get("A_MEAN")
        self.b_std = config.get("B_STD")
        self.b_mean = config.get("B_MEAN")

    def forward(self, stack: Tensor, rfv: Tensor) -> Tuple[Tensor, Tensor]:
        """forward(x) = x + sqrt(a*x + b) * N(0,1), where a and b are calibrated parameters

        Args:
            stack (Tensor): the noise-free stack of light field views normalized to [0,1]
            rfv (Tensor): the noise-free refocused volume normalized to [0,1]

            note: ideally we would perform the backprojection on the raw LF views with the noise added
            to attain the RFV. we simplify this by just adding the same noise to both, and dividing the std by the
            sqrt of the number of views being added to form the RFV, like an "denoising" approach.
        Returns:
            Tuple[Tensor, Tensor]: stack and rfv with poisson-gaussian noise added
        """
        a = torch.randn(1) * self.a_std + self.a_mean
        b = torch.randn(1) * self.b_std + self.b_mean
        recip_sqrt_num_views = 1 / torch.sqrt(
            stack.shape[1]
        )  # first dim is batch size, 2nd is channels/num views

        stack += torch.sqrt(torch.clamp(a * stack + b, min=0)) * torch.randn(
            stack.shape
        )
        rfv += (
            torch.sqrt(torch.clamp(a * rfv + b, min=0))
            * torch.randn(rfv.shape)
            * recip_sqrt_num_views
        )

        return stack, rfv
