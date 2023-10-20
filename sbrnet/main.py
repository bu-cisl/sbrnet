import logging
from config_loader import load_config
from data_loader import DataLoader
from model import SimpleNet
from trainer import Trainer

# Configure logging to write log messages to a file
log_file_path = "my_log_file.log"
logging.basicConfig(filename=log_file_path, level=logging.DEBUG)

logger = logging.getLogger(__name)

if __name__ == "__main__":
    config = load_config("config.yaml")

    data_loader = DataLoader(config)
    model = SimpleNet(config)
    trainer = Trainer(model, data_loader.load_data(config["data_folder"]), config)

    logger.info("Starting training...")
    trainer.train()

    logger.info("Training complete.")
