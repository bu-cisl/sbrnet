from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor, Module


class PoissonGaussianNoiseModel(Module):
    """Poisson-Gaussian noise model for CM2 sensor: https://ieeexplore.ieee.org/document/4623175"""

    # from calibrated CM2 SONY imx-226 sensor on [0,1] normalized data
    A_STD = 5.7092e-5
    A_MEAN = 1.49e-4
    B_STD = 2.7754e-6
    B_MEAN = 5.41e-6

    def __init__(self, a_std=A_STD, a_mean=A_MEAN, b_std=B_STD, b_mean=B_MEAN):
        super().__init__()

        self.a_std = a_std
        self.a_mean = a_mean
        self.b_std = b_std
        self.b_mean = b_mean

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

        stack += stack.sqrt(torch.clamp(a * stack + b, min=0)) * torch.randn(
            stack.shape
        )
        rfv += (
            rfv.sqrt(torch.clamp(a * rfv + b, min=0))
            * torch.randn(rfv.shape)
            * 0.333  # hardcoding for now for performance optimization
        )

        return stack, rfv
