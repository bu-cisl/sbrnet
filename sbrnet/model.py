import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from config_loader import load_config

config = load_config("config.yaml")

SQRT2 = 1.414
RSQRT2 = 1 / SQRT2 # NOTE: multiplication is faster than division for floats


class SBRNet(nn.Module):
    def __init__(self, config) -> None:
        super(SBRNet, self).__init__()

        # UNet backbone is deprecated
        if config["backbone"] == "resnet":
            self.view_synthesis_branch = ResNetCM2NetBlock()


class ResConnection(nn.Sequential):
    def forward(self, data: Tensor, scale = RSQRT2) -> Tensor:
        return (super().forward(data) + data) * scale

# ResNet backbone
class ResBlock(ResConnection):
    def __init__(self, channels=config["resnet_channels"]):

        self.channels = channels # does this really need to be stored?

        # allocate conv layers
        # NOTE: I would generally allow for the number of convs to be a parameter
        #       personally I don't like the rigid nature of forcing only 2 convs
        conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

        # initalize conv layers
        for conv in [conv1, conv2]:
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")

        super(ResBlock, self).__init__(
            conv1,
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            conv2,
            nn.BatchNorm2d(channels),
        )


class ResNetCM2NetBlock(ResConnection):
    def __init__(self, inchannels: int, numblocks: int, outchannels: int = 48):
        """_summary_

        Args:
            inchannels (int): depends on the branch. if view synthesis branch, equals the number of views. if rfv branch, equals the number of slices in the volume.
            numblocks (int): _description_
            outchannels (int, optional): _description_. Defaults to 48.
        """

        conv1 = nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1)
        conv2 = nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1)

        # initalize conv layers
        for conv in [conv1, conv2]:
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")

        resblocks = [ResBlock() for i in range(numblocks)]

        super(ResNetCM2NetBlock, self).__init__(
            conv1, # no activation or batch norm?
            *resblocks,
            conv2 # no activation or batch norm?
        )

        # dont think this these are necessary to keep
        self.inchannels  = inchannels
        self.outchannels = outchannels
        self.numblocks   = numblocks
