# Importing modules
import torch
import torch.nn as nn
import torch.nn.functional as F
# Importing PyTorch timm module for pre-trained Efficient Net architecture
import timm


# Type imports for type annotations
from torch import Tensor
from torch.nn import Module, Conv2d, Sequential

# Initializing square root tensor of 1/2
RSQRT2 = torch.sqrt(torch.tensor(0.5)).item()


# Defining EfficientNet Architecture
class EfficientNetBlock(Module):
    def __init__(self, config, branch: str) -> None:
        super(EfficientNetBlock, self).__init__()
        # Assuming 'efficientnet_b0' is desired, can be changed according to needs
        model_name = config.get('efficientnet_model', 'efficientnet_b0')
        
        # Load a pre-trained EfficientNet model
        self.efficient_net = timm.create_model(model_name, pretrained=True, num_classes=0)

        # Additional layers to match the output dimensions expected by SBRNet
        if branch == "view_synthesis":
            in_channels = config["num_lf_views"]
        elif branch == "refinement":
            in_channels = config["num_rfv_layers"]
        else:
            raise ValueError(f"Unknown branch: {branch}.")
            
        self.adapt_conv = nn.Conv2d(in_channels, self.efficient_net.num_features, kernel_size=1)
        self.output_conv = nn.Conv2d(self.efficient_net.num_features, config["num_gt_layers"], kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.adapt_conv(x)
        x = self.efficient_net(x)
        x = self.output_conv(x)
        return x


# Defining SBREfficientNet model class
class SBREfficientNet(Module):
    """ """
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        if config["backbone"] == "efficientnet":
            self.view_synthesis_branch = EfficientNetBlock(config, "view_synthesis")

            self.rfv_branch = EfficientNetBlock(config, "refinement")
        else:
            raise ValueError(f"Unknown backbone: {config['backbone']}. Only 'efficientnet' is supported now.")

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
        # Process each branch
        lf_view_features = self.view_synthesis_branch(lf_view_stack)
        rfv_features = self.rfv_branch(rfv)
        
        # Combine the outputs from the two branches and normalize
        combined_features = (lf_view_features + rfv_features) * RSQRT2
        
        # Apply the end convolution
        output = self.end_conv(combined_features)
        return output