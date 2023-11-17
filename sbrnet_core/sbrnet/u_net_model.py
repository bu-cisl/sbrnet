# Importing modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# Type imports
from torch import Tensor
from torch.nn import Module, Conv2d, BatchNorm2d

# Initializing square root tensor of 1/2
RSQRT2 = torch.sqrt(torch.tensor(0.5)).item()


# Defining SBRUNet class
class SBRUNet(nn.Module):
    """SBRNet with a U-Net backbone"""
    def __init__(self, config) -> None:
        """
        Initialize the SBRUNet with the given configuration.

        Args:
            config (dict): Configuration dictionary specifying network options and parameters.
        """
        super().__init__()
        self.config = config

        # Verify the backbone specified in the configuration is U-Net.
        if config["backbone"] == "unet":
            self.view_synthesis_branch = UNetBlock(config, "view_synthesis")
            self.rfv_branch = UNetBlock(config, "refinement")
        else:
            raise ValueError(f"Unsupported backbone: {config['backbone']}. Expected 'unet'.")
        
        # Define the final convolution layer that combines the outputs from both branches.
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
        """ Forward pass of the network that processes input through both branches and combines them."""
        return self.end_conv((self.view_synthesis_branch(lf_view_stack) + self.rfv_branch(rfv)) * RSQRT2)


# Creating DoubleConv class
class DoubleConv(nn.Module):
    """Module  performs two consecutive sets of convolution, batch normalization, and ReLU activation
    (convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):        
        """
        Initialize the DoubleConv module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()

        # Sequentially apply two sets of Conv, BN, ReLU layers.
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Pass the input through the double convolution layers.
        return self.double_conv(x)
    

class UNetEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Define the convolutional layers and downsampling
        self.conv = DoubleConv(in_channels, out_channels)
        self.down = nn.MaxPool2d(2)

    def forward(self, x):
        # Apply convolutional layers and return both output and input for skip connections
        conv_output = self.conv(x)
        return self.down(conv_output), conv_output

class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Define upsampling and convolutional layers
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.up(x)
        # Concatenate skip connection
        x = torch.cat([x, skip_connection], dim=1)
        return self.conv(x)
    

# Creating UNetBlock Class
class UNetBlock(nn.Module):
    """ 
    U-Net block that performs downsampling followed by upsampling and concatenates 
    the features with the original input to form a U-Net style block. """
    def __init__(self, config, branch: str):
        """
        Initialize the U-Net block.

        Args:
            config (dict): Configuration dictionary specifying network options.
            branch (str): The branch type for which the block is being created, either 'view_synthesis' or 'refinement'.
        """
        super(UNetBlock, self).__init__()

        # Determine the input channels based on the branch type
        if branch == "view_synthesis":
            in_channels = config["num_lf_views"]
        elif branch == "refinement":
            in_channels = config["num_rfv_layers"]
        else:
            raise ValueError(f"Unknown branch: {branch}. Only 'view_synthesis' and 'refinement' are supported.")
        
         # Number of output channels based on configuration
        out_channels = config["num_gt_layers"]

        # The encoder performs the initial convolution operations.
        self.encoder = DoubleConv(in_channels, out_channels)
        # The decoder performs the convolution operations after upsampling.
        self.decoder = DoubleConv(out_channels * 2, out_channels)

        # Downsample the feature maps by a factor of 2 using Max Pooling.
        self.down = nn.MaxPool2d(2)
        # Upsample the feature maps by a factor of 2 using Transpose Convolution.
        self.up = nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=2, stride=2)

    def forward(self, x):
        # Apply the encoder to obtain the feature maps.
        enc = self.encoder(x)
        # Save the feature maps for skip connection (before downsampling)
        skip_connection = enc
        # Downsample the feature maps.
        enc = self.down(enc)
        # Upsample the feature maps.
        dec = self.up(enc)
        # Concatenate the upsampled feature maps with the saved skip connection
        # Ensure the spatial dimensions match for concatenation
        crop_size = (skip_connection.shape[2] - dec.shape[2]) // 2
        skip_connection = skip_connection[:, :, crop_size:skip_connection.shape[2]-crop_size, crop_size:skip_connection.shape[3]-crop_size]
        # Concatenate along the channel dimension
        dec = torch.cat([dec, skip_connection], dim=1)
        # Apply the decoder.
        return self.decoder(dec)
    
