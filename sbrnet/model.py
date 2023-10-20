import torch
import torch.nn as nn
import torch.nn.functional as F
from config_loader import load_config

config = load_config("config.yaml")

SQRT2 = 1.414


class SBRNet(nn.Module):
    def __init__(self, config) -> None:
        super(SBRNet, self).__init__()

        # UNet backbone is deprecated
        if config["backbone"] == "resnet":
            self.view_synthesis_branch = ResNetCM2NetBlock()


# ResNet backbone
class ResBlock(nn.Module):
    def __init__(self, channels=config["resnet_channels"]):
        super(ResBlock, self).__init__()
        self.channels = channels

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x1):
        x1 = self.conv1(x1)
        x1 = F.relu(self.bn1(x1), inplace=True)
        x1 = self.conv2(x1)
        return self.bn2(x1)


class ResNetCM2NetBlock(nn.Module):
    def __init__(self, inchannels: int, numblocks: int, outchannels: int = 48):
        """_summary_

        Args:
            inchannels (int): depends on the branch. if view synthesis branch, equals the number of views. if rfv branch, equals the number of slices in the volume.
            numblocks (int): _description_
            outchannels (int, optional): _description_. Defaults to 48.
        """
        super(ResNetCM2NetBlock, self).__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.numblocks = numblocks

        self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        self.resblocks = nn.ModuleList([ResBlock() for i in range(numblocks)])
        self.conv2 = nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = torch.clone(x0)
        for _, m in enumerate(self.resblocks):
            x1 = (m(x1) + x1) / SQRT2  # short residual connection
        x1 = (x1 + x0) / SQRT2  # long residual connection
        return self.conv2(x1)


# class ResNetCM2Net(nn.Module):
#     def __init__(self, numBlocks, stackchannels=numViews, rfvchannels=24, outchannels=24):
#         super(ResNetCM2Net, self).__init__()

#         self.stackpath = CM2NetBlock(stackchannels, numblocks=numBlocks)
#         self.rfvpath = CM2NetBlock(rfvchannels, numblocks=numBlocks)
#         self.endconv = nn.Conv2d(outchannels*2, outchannels, kernel_size=3, padding=1)
#         nn.init.kaiming_normal_(self.endconv.weight, nonlinearity='relu')

#     def forward(self, stack, rfv):
#         return self.endconv((self.stackpath(stack) + self.rfvpath(rfv)) / SQRT2) # branch fusion
