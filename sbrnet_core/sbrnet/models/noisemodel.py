from typing import Tuple
from functools import cached_property
import torch
from torch import Tensor
from torch.nn import Module
from pandas import read_parquet


class PoissonGaussianNoiseModel(Module):
    """Poisson-Gaussian noise model for CM2 sensor: https://ieeexplore.ieee.org/document/4623175"""

    def __init__(self, config: dict):
        super().__init__()
        df = read_parquet(config["dataset_pq"])
        self.num_views = df.iloc[0].num_views
        self.a_mean = config.get("A_MEAN")
        self.b_mean = config.get("B_MEAN")

    @cached_property
    def recip_sqrt_num_views(self) -> torch.Tensor:
        return 1 / torch.sqrt(torch.tensor(self.num_views))

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

        recip_sqrt_num_views = self.recip_sqrt_num_views.to(stack.device)

        stack += torch.sqrt(
            torch.clamp(self.a_mean * stack + self.b_mean, min=0)
        ) * torch.randn(stack.shape).to(stack.device)

        rfv += (
            torch.sqrt(torch.clamp(self.a_mean * rfv + self.b_mean, min=0))
            * torch.randn(rfv.shape).to(rfv.device)
            * recip_sqrt_num_views
        )
        return stack, rfv
