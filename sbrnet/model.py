import torch
import torch.nn as nn
import torch.nn.functional as F

from config_loader import load_config

# types imports
from torch import Tensor
from torch.nn import Module, Conv2d, Sequential

config = load_config("config.yaml")

RSQRT2 = torch.sqrt(0.5).item() # NOTE: multiplication is faster than division for floats

class SBRNet(Module):
    def __init__(self, config) -> None:
        super(SBRNet, self).__init__()

        # UNet backbone is deprecated
        if config["backbone"] == "resnet":
            self.view_synthesis_branch: Module = ResNetCM2NetBlock()
        
        self.init_convs()
    
    def init_convs(self) -> None:
        def init_fn(mod: Module) -> None:
            if isinstance(mod, Conv2d):
                nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")
        
        # initializes all Conv2d
        self.view_synthesis_branch.apply(init_fn)


class ResConnection(Sequential):
    def forward(self, data: Tensor, scale: float = RSQRT2) -> Tensor:
        return (super().forward(data) + data) * scale


# ResNet backbone
class ResBlock(ResConnection):
    def __init__(self, channels: int = config["resnet_channels"]) -> None:

        super(ResBlock, self).__init__(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )


class ResNetCM2NetBlock(ResConnection):
    def __init__(self, inchannels: int, numblocks: int, outchannels: int = 48) -> None:
        """_summary_

        Args:
            inchannels (int): depends on the branch. if view synthesis branch, equals the number of views. if rfv branch, equals the number of slices in the volume.
            numblocks (int): _description_
            outchannels (int, optional): _description_. Defaults to 48.
        """

        super(ResNetCM2NetBlock, self).__init__(
            nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1), # no activation or batch norm?
            *(ResBlock() for i in range(numblocks)),
            nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1) # no activation or batch norm?
        )
