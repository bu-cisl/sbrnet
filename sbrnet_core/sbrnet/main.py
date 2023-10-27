import logging
import argparse
from datetime import datetime  # Import datetime module
from torch import compile

from sbrnet_core.sbrnet.model import SBRNet
from sbrnet_core.config_loader import load_config
from sbrnet_core.sbrnet.trainer import Trainer

### use only in SCC interactive mode
# import os

# # go to powershell and do nvidia-smi to see which GPU you are assigned
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
###

# Get the current timestamp as a string
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define the log file path with the timestamp
log_file_path = f"/projectnb/tianlabdl/jalido/sbrnet_proj/.log/logging/sbrnet_train_{current_time}.log"

# Configure logging to write log messages to the file
logging.basicConfig(filename=log_file_path, level=logging.INFO)

logger = logging.getLogger(__name__)


def main(config_file):
    config = load_config(config_file)

    model = compile(SBRNet(config))

    trainer = Trainer(model, config)

    logger.info("Starting training...")

    trainer.train()

    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SBRNet using a config file.")
    parser.add_argument(
        "config_file", help="Path to the config file (e.g., config.yaml)"
    )

    args = parser.parse_args()

    main(args.config_file)
