# Importing modules
import torch
import torch.nn as nn
import torch.nn.functional as F
# Importing PyTorch timm module for pre-trained Dense Net architecture
import timm


# Type imports for type annotations
from torch import Tensor
from torch.nn import Module, Conv2d, Sequential

# Initializing square root tensor of 1/2
RSQRT2 = torch.sqrt(torch.tensor(0.5)).item()

class SBRDenseNet(Module):
    def __init__(self, config) -> None:
        super(SBRDenseNet, self).__init__()
        self.config = config
        
        if config["backbone"] != "densenet":
            raise ValueError(f"Unsupported backbone: {config['backbone']}. Expected 'densenet'.")
        
        self.view_synthesis_branch = DenseNetBlock(config, "view_synthesis")
        self.rfv_branch = DenseNetBlock(config, "refinement")
        
        # End convolution to combine the outputs from both branches
        self.end_conv = nn.Conv2d(
            config["num_gt_layers"] * 2,
            config["num_gt_layers"],
            kernel_size=1  # Changed to kernel size 1 to keep dimensions unchanged
        )
    
    def forward(self, lf_view_stack: Tensor, rfv: Tensor) -> Tensor:
        lf_view_features = self.view_synthesis_branch(lf_view_stack)
        rfv_features = self.rfv_branch(rfv)
        
        combined_features = (lf_view_features + rfv_features) * RSQRT2
        
        output = self.end_conv(combined_features)
        return output
    
    
# Creating Dense Net Block architecture class
class DenseNetBlock(Module):
    def __init__(self, config, branch: str) -> None:
        super(DenseNetBlock, self).__init__()
        # Assuming 'densenet121' is desired, can be changed according to needs
        model_name = config.get('densenet_model', 'densenet121')
        
        # Load a pre-trained DenseNet model without the classification head
        self.dense_net = timm.create_model(model_name, pretrained=True, features_only=True)
        
        # Define the input adaptation and output convolution layers
        if branch == "view_synthesis":
            in_channels = config["num_lf_views"]
        elif branch == "refinement":
            in_channels = config["num_rfv_layers"]
        else:
            raise ValueError(f"Unknown branch: {branch}.")
            
        self.adapt_conv = nn.Conv2d(in_channels, self.dense_net.num_features, kernel_size=1)
        self.output_conv = nn.Conv2d(self.dense_net.num_features, config["num_gt_layers"], kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.adapt_conv(x)
        features = self.dense_net(x)
        x = torch.cat(features, 1)  # Concatenate the feature maps from all dense blocks
        x = self.output_conv(x)
        return x
