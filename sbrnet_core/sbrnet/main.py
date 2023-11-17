# Importing Modules
import logging
import argparse
from datetime import datetime
from torch import compile

# Importing Classes
from sbrnet_core.sbrnet.res_net_model import SBRResNet
from sbrnet_core.sbrnet.dense_net_model import SBRDenseNet
from sbrnet_core.sbrnet.efficient_net_model import SBREfficientNet
from sbrnet_core.sbrnet.u_net_model import SBRUNet
from sbrnet_core.config_loader import load_config
from sbrnet_core.sbrnet.trainer import Trainer

### use only in SCC interactive mode
# import os
## Setting environment variables
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(config_file):
    """
    The main function that loads configuration file, initializes model, and starts training.

    Args:
        config_file (str): The path to the YAML configuration file.
    """

    # Load the configuration settings from the provided YAML config file 
    config = load_config(config_file)

    # Determine the backbone from the config
    backbone = config["backbone"]

    # Get the current timestamp as a string
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Define the log file path with the timestamp
    log_file_path = f"/projectnb/tianlabdl/nrabines/sbrnet/.log/logging/sbrnet_train_{current_time}.log"

    # Configure logging to write log messages to the file
    logging.basicConfig(filename=log_file_path, level=logging.INFO)
    logger = logging.getLogger(__name__)  # Create a logger instance for this module

    # Initialize SBRNet model with configuration on specified backbone in config file
    if backbone == "resnet":
        model = SBRResNet(config)
    elif backbone == "densenet":
        model = SBRDenseNet(config)
    elif backbone == "efficientnet":
        model = SBREfficientNet(config)
    elif backbone == "unet":
        model = SBRUNet(config)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    
    # Compiling model with specified CNN architecture
    model = compile(model)

    # Create a Trainer instance with the model and configuration
    trainer = Trainer(model, config)

    # Log the start of training
    logger.info("Starting training...")

    # Start the training process
    trainer.train()

    # Log the completion of training
    logger.info("Training complete.")


if __name__ == "__main__":
    # Set up an argument parser to allow for command-line arguments
    parser = argparse.ArgumentParser(description="Train SBRNet using config file.")

    # Arguement for specifying the directory for config.yaml
    # Note: make sure to specify correct .yaml file when running code for desired architecture
    parser.add_argument("config_file", help="Path to the config file (e.g., config.yaml)")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the provided config file path
    main(args.config_file)
