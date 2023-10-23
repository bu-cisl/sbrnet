import logging

from sbrnet_core.sbrnet.model import SBRNet
from sbrnet_core.sbrnet.config_loader import load_config

# from sbrnet_core.sbrnet.dataset import DataLoader
from sbrnet_core.sbrnet.model import SBRNet
from sbrnet_core.sbrnet.trainer import Trainer

# Configure logging to write log messages to a file
log_file_path = "my_log_file.log"
logging.basicConfig(filename=log_file_path, level=logging.DEBUG)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    config = load_config("config.yaml")

    model = SBRNet(config)

    trainer = Trainer(model, config)

    logger.info("Starting training...")

    trainer.train()

    logger.info("Training complete.")
