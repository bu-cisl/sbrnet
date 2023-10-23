import torch
import torch.nn as nn
import torch.nn.functional as F

# types imports
from torch import Tensor
from torch.nn import Module, Conv2d, Sequential

RSQRT2 = torch.sqrt(
    0.5
).item()  # NOTE: multiplication is faster than division for floats


class SBRNet(Module):
    def __init__(self, config) -> None:
        super(SBRNet, self).__init__()

        # UNet backbone is deprecated
        if config["backbone"] == "resnet":
            self.view_synthesis_branch: Module = ResNetCM2NetBlock(config, "")
            self.rfv_branch: Module = ResNetCM2NetBlock(config)
        else:
            raise ValueError(
                f"Unknown backbone: {config['backbone']}. Only 'resnet' is supported."
            )
        self.end_conv: Module = nn.Conv2d(
            config.get("num_gt_layers") * 2,
            config.get("num_gt_layers"),
            kernel_size=3,
            padding=1,
        )
        self.init_convs()

    def init_convs(self) -> None:
        def init_fn(mod: Module) -> None:
            if isinstance(mod, Conv2d):
                nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")

        # initializes all Conv2d
        self.view_synthesis_branch.apply(init_fn)

    def forward(self, lf_view_stack: Tensor, rfv: Tensor) -> Tensor:
        return self.view_synthesis_branch(lf_view_stack) + self.rfv_branch(rfv)


class ResConnection(Sequential):
    def forward(self, data: Tensor, scale: float = RSQRT2) -> Tensor:
        return (super().forward(data) + data) * scale


# ResNet backbone
class ResBlock(ResConnection):
    def __init__(self, channels: int) -> None:
        super(ResBlock, self).__init__(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )


class ResNetCM2NetBlock(ResConnection):
    def __init__(self, config, branch: str) -> None:
        if branch == "view_synthesis":
            inchannels = config["num_lf_views"]
        elif branch == "refinement":
            inchannels = config["num_rfv_layers"]
        else:
            raise ValueError(
                f"Unknown branch: {branch}. Only 'view_synthesis' and 'refinement' are supported."
            )

        numblocks = config["num_resblocks"]
        outchannels = config["num_gt_layers"]
        super(ResNetCM2NetBlock, self).__init__(
            nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1),
            *(ResBlock(channels=outchannels * 2) for i in range(numblocks)),
            nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1),
        )
