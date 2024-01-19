import logging
from pandas import read_parquet
import torch
import torch.nn as nn
import torch.nn.functional as F

# types imports
from torch import Tensor
from torch.nn import Module, Conv2d, Sequential

RSQRT2 = torch.sqrt(torch.tensor(0.5)).item()
logger = logging.getLogger(__name__)


class SimpleLayer(nn.Module):
    def __init__(self, config):
        super(SimpleLayer, self).__init__()
        self.layer = nn.Sequential(
            *[
                nn.Conv2d(
                    config.get("num_gt_layers") * 2,
                    config.get("num_gt_layers") * 2,
                    kernel_size=3,
                    padding=1,
                )
                if _ < config.get("num_head_layers") - 1
                else nn.Conv2d(
                    config.get("num_gt_layers") * 2,
                    config.get("num_gt_layers"),
                    kernel_size=3,
                    padding=1,
                )
                for _ in range(config.get("num_head_layers"))
            ]
        )

    def forward(self, x):
        return self.layer(x)


class QuantileLayer(nn.Module):
    def __init__(self, config):
        super(QuantileLayer, self).__init__()
        n_channels_middle = config.get("num_gt_layers") * 2
        n_channels_out = config.get("num_gt_layers")

        self.lower = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        config.get("num_gt_layers") * 2,
                        config.get("num_gt_layers") * 2,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(),
                )

                for _ in range(config.get("num_head_layers") - 1)
            ],
            nn.Conv2d(
                config.get("num_gt_layers") * 2,
                config.get("num_gt_layers"),
                kernel_size=3,
                padding=1,
            ),
        )
        self.prediction = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        config.get("num_gt_layers") * 2,
                        config.get("num_gt_layers") * 2,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(),
                )

                for _ in range(config.get("num_head_layers") - 1)
            ],
            nn.Conv2d(
                config.get("num_gt_layers") * 2,
                config.get("num_gt_layers"),
                kernel_size=3,
                padding=1,
            ),
        )
        self.upper = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        config.get("num_gt_layers") * 2,
                        config.get("num_gt_layers") * 2,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(),
                )

                for _ in range(config.get("num_head_layers") - 1)
            ],
            nn.Conv2d(
                config.get("num_gt_layers") * 2,
                config.get("num_gt_layers"),
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, x):
        return torch.cat([self.lower(x), self.prediction(x), self.upper(x)], dim=1)


# class QuantileLayer(nn.Module):
#     def __init__(self, config):
#         super(QuantileLayer, self).__init__()
#         # expand from the trunk to 3 heads, each head with (num_gt_layers * 2) channels
#         self.expansion = nn.Conv2d(
#             config.get("num_gt_layers") * 2,
#             config.get("num_gt_layers") * 2 * 3,
#             kernel_size=3,
#             padding=1,
#         )

#         #each head has their own group of conv kernels with groups=3
#         self.multihead = nn.Sequential(
#             *[
#                 nn.Sequential(
#                     nn.Conv2d(
#                         config.get("num_gt_layers") * 2 * 3,
#                         config.get("num_gt_layers") * 2 * 3,
#                         kernel_size=3,
#                         padding=1,
#                         groups=3,
#                     ),
#                     nn.ReLU(),
#                 )
#                 # minimum 2 layers for expansion and contraction.
#                 # any more would be here in the middle.
#                 for _ in range(config.get("num_head_layers") - 2)
#             ]
#         )
#         # single conv layer to shrink each head's (num_gt_layers * 2) channels
#         # to (num_gt_layers) for the final reconstruction
#         self.contraction = nn.Conv2d(
#             config.get("num_gt_layers") * 2 * 3,
#             config.get("num_gt_layers") * 3,
#             kernel_size=3,
#             padding=1,
#             groups=3,
#         )

#     def forward(self, x):
#         x = self.expansion(x)
#         x = self.multihead(x)
#         # x = (self.multihead(x) + x) * RSQRT2
#         x = self.contraction(x)
#         return x


class LastLayer(nn.Module):
    def __init__(self, config):
        super(LastLayer, self).__init__()

        self.last_layer = config.get("last_layer", "quantile_heads")

        if self.last_layer == "quantile_heads":
            self.layer = QuantileLayer(config)
            logger.info("Using quantile heads")
        else:
            self.layer = SimpleLayer(config)
            logger.info("Using simple single head")

    def forward(self, x):
        return self.layer(x)


# class SBRNet(Sequential):
#     def __init__(self, config):
#         super(SBRNet, self).__init__(SBRNetTrunk(config), LastLayer(config))
class SBRNet(nn.Module):
    def __init__(self, config: dict):
        super(SBRNet, self).__init__()
        self.config = config
        self.trunk = SBRNetTrunk(config)
        self.last_layer = LastLayer(config)

    def forward(self, lf_view_stack: Tensor, rfv: Tensor) -> Tensor:
        return self.last_layer(self.trunk(lf_view_stack, rfv))


class SBRNetTrunk(Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        # UNet backbone is deprecated
        if config["backbone"] == "resnet":
            self.view_synthesis_branch = ResNetCM2NetBlock(config, "view_synthesis")

            self.rfv_branch = ResNetCM2NetBlock(config, "refinement")
        else:
            raise ValueError(
                f"Unknown backbone: {config['backbone']}. Only 'resnet' is supported."
            )

        self.init_convs()

    def init_convs(self) -> None:
        def init_fn(mod: Module) -> None:
            if isinstance(mod, Conv2d):
                weight_init = self.config.get("weight_init", "kaiming_normal")
                if weight_init == "kaiming_normal":
                    nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")
                elif weight_init == "xavier_normal":
                    nn.init.xavier_normal_(mod.weight)
                else:
                    raise ValueError(
                        f"Unsupported weight initialization method: {weight_init}"
                    )

        # initializes all Conv2d
        self.view_synthesis_branch.apply(init_fn)
        self.rfv_branch.apply(init_fn)

    def forward(self, lf_view_stack: Tensor, rfv: Tensor) -> Tensor:
        return (
            self.view_synthesis_branch(lf_view_stack) + self.rfv_branch(rfv)
        ) * RSQRT2


class ResConnection(Sequential):
    def forward(self, data: Tensor, scale: float = RSQRT2) -> Tensor:
        return (super().forward(data) + data) * scale


# # ResNet backbone by mitch
# class ResBlock(Sequential):
#     def __init__(self, channels: int) -> None:
#         super(ResBlock, self).__init__(
#             nn.BatchNorm2d(channels),
#             ResConnection(
#                 nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
#                 nn.ReLU(True),
#                 nn.BatchNorm2d(channels),
#                 nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
#             ),
#         )


# og resnet backbone
class ResBlock(ResConnection):
    def __init__(self, channels: int) -> None:
        super(ResBlock, self).__init__(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )


class ResNetCM2NetBlock(Sequential):
    def __init__(self, config, branch: str) -> None:
        df = read_parquet(config["dataset_pq"])

        if branch == "view_synthesis":
            inchannels = df.iloc[0].num_views
        elif branch == "refinement":
            inchannels = config["num_rfv_layers"]
        else:
            raise ValueError(
                f"Unknown branch: {branch}. Only 'view_synthesis' and 'refinement' are supported."
            )

        numblocks = config["num_resblocks"]
        outchannels = config["num_gt_layers"]
        # # mitchell's idea.
        # super(ResNetCM2NetBlock, self).__init__(
        #     nn.Conv2d(inchannels, outchannels * 2, kernel_size=3, padding=1),
        #     *(ResBlock(channels=outchannels * 2) for _ in range(numblocks)),
        # )

        # # og
        super(ResNetCM2NetBlock, self).__init__(
            nn.Conv2d(inchannels, outchannels * 2, kernel_size=3, padding=1),
            ResConnection(
                *(ResBlock(channels=outchannels * 2) for _ in range(numblocks))
                # nn.Conv2d(outchannels * 2, outchannels * 2, kernel_size=3, padding=1),
            ),
        )


# debugging
if __name__ == "__main__":
    config = {
        "backbone": "resnet",
        "num_gt_layers": 24,
        "num_rfv_layers": 24,
        "num_resblocks": 2,
        "num_head_layers": 2,
        "weight_init": "kaiming_normal",
        "last_layer": "quantile_heads",
        "dataset_pq": "/ad/eng/research/eng_research_cisl/jalido/sbrnet/data/training_data/UQ/15/metadata.pq",
    }
    stack = torch.randn(1, 9, 32, 32)
    rfv = torch.randn(1, 24, 32, 32)
    model = SBRNet(config)
    out = model(stack, rfv)
    print(out.shape)

    slice_q_lo = slice(0, config["num_gt_layers"])
    slice_q_hi = slice(config["num_gt_layers"], config["num_gt_layers"] * 2)
    slice_point = slice(config["num_gt_layers"] * 2, None)

    from sbrnet_core.sbrnet.losses.quantile_loss import PinballLoss

    q_lo_loss = PinballLoss(quantile=0.10, reduction="mean")
    print("SKice", slice_q_lo)
    print(out[:, slice_q_lo, :, :].shape, rfv.shape)
    a = q_lo_loss(out[:, slice_q_lo, :, :], rfv)
    print(a)
