# Importing modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# types imports
from torch import Tensor
from torch.nn import Module, Conv2d, Sequential

# Initializing tensor for the square root of 1/2
RSQRT2 = torch.sqrt(torch.tensor(0.5)).item()


# Creating SBRResNet class
class SBRResNet(Module):
    """SBRNet with a ResNet backbone"""
    def __init__(self, config) -> None:
        """ 
        Initializes the network with the given configuration. 
        Args: 
            config (dict): Configuration dictionary specifying network options. 
        """
        super().__init__()
        self.config = config

        # UNet backbone is deprecated
        if config["backbone"] == "resnet":
            self.view_synthesis_branch = ResNetCM2NetBlock(config, "view_synthesis")
            self.rfv_branch = ResNetCM2NetBlock(config, "refinement")
        else:
            raise ValueError(f"Unknown backbone: {config['backbone']}. Only 'resnet' is supported.")
        
        # Final convolution layer that combines the output from both branches
        self.end_conv: Module = nn.Conv2d(
            config.get("num_gt_layers") * 2,
            config.get("num_gt_layers"),
            kernel_size=3,
            padding=1,
        )

        # Initialize weights of convolution layers
        self.init_convs()


    def init_convs(self) -> None:
        """Initializes the convolution layers with weights according to the specified method in the config."""
        def init_fn(mod: Module) -> None:
            # Initialize weights only for Conv2d modules
            if isinstance(mod, Conv2d):
                # Get weight initialization method from config, default to 'kaiming_normal'
                weight_init = self.config.get("weight_init", "kaiming_normal")
                if weight_init == "kaiming_normal":
                    nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")
                elif weight_init == "xavier_normal":
                    nn.init.xavier_normal_(mod.weight)
                else:
                    raise ValueError(f"Unsupported weight initialization method: {weight_init}")

        # Apply the initialization function to all Conv2d layers in both branches
        self.view_synthesis_branch.apply(init_fn)
        self.rfv_branch.apply(init_fn)

    def forward(self, lf_view_stack: Tensor, rfv: Tensor) -> Tensor:
        """ Forward pass of network processes input through both branches and combines them."""
        return self.end_conv((self.view_synthesis_branch(lf_view_stack) + self.rfv_branch(rfv)) * RSQRT2)


# Creating ResConnection class
class ResConnection(Sequential):
    """Custom ResNet connection that applies scaling after adding the input to the output of a block."""
    def forward(self, data: Tensor, scale: float = RSQRT2) -> Tensor:
        """ Overrides forward pass to apply a residual connection with scaling."""
        return (super().forward(data) + data) * scale


# ResNet backbone
class ResBlock(ResConnection):
    """A standard ResNet block that performs two sequences of convolution, batch normalization, and ReLU activation."""
    def __init__(self, channels: int) -> None:
        """
        Initializes the ResBlock with a set number of channels for convolution layers.

        Args:
            channels (int): Number of channels for the convolution layers.
        """
        super(ResBlock, self).__init__(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )


class ResNetCM2NetBlock(Sequential):
    """ 
    A sequential block representing a portion of a ResNet architecture, which can be used 
    for either view synthesis or refinement in the SBRNet architecture. 
    """
    def __init__(self, config, branch: str) -> None:
        """
        Initializes a block with a series of ResBlocks and convolution layers according to the configuration.

        Args:
            config (dict): Configuration dictionary specifying network options.
            branch (str): The branch type for which the block is being created, either 'view_synthesis' or 'refinement'.
        """
        # Determine the input channels based on the branch type
        if branch == "view_synthesis":
            inchannels = config["num_lf_views"]
        elif branch == "refinement":
            inchannels = config["num_rfv_layers"]
        else:
            raise ValueError(f"Unknown branch: {branch}. Only 'view_synthesis' and 'refinement' are supported.")

        # Number of ResBlocks and output channels based on configuration
        numblocks = config["num_resblocks"]
        outchannels = config["num_gt_layers"]

        # Initialize the block with input convolution, ResBlocks, and output convolution
        super(ResNetCM2NetBlock, self).__init__(
            nn.Conv2d(inchannels, outchannels * 2, kernel_size=3, padding=1),
            ResConnection(*(ResBlock(channels=outchannels * 2) for _ in range(numblocks))),
            nn.Conv2d(outchannels * 2, outchannels * 2, kernel_size=3, padding=1)
        )
