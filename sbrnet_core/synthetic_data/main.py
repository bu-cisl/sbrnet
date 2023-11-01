import logging
import argparse
from datetime import datetime  # Import datetime module
import time

from sbrnet_core.config_loader import load_config
from sbrnet_core.synthetic_data.generate_synthetic_data_v1 import make_synthetic_dataset

# Get the current timestamp as a string
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define the log file path with the timestamp
log_file_path = f"/projectnb/tianlabdl/jalido/sbrnet_proj/.log/logging/sbrnet_synthetic_data_{current_time}.log"

# Configure logging to write log messages to the file
logging.basicConfig(filename=log_file_path, level=logging.DEBUG)

logger = logging.getLogger(__name__)


def main(config_file):
    config = load_config(config_file)

    logger.info("Starting synthetic data generation...")
    logger.info(f"Using ray: {config['use_ray']}")
    start_time = time.time()
    make_synthetic_dataset(config)
    logger.info("Synthetic data generation complete.")
    logger.info(
        f"With use_ray set to {config['use_ray']}, total time: {time.time() - start_time} seconds."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic data using a config file."
    )
    parser.add_argument(
        "config_file", help="Path to the config file (e.g., config.yaml)"
    )

    args = parser.parse_args()

    main(args.config_file)
