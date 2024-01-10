import logging
import argparse
from datetime import datetime
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


def main(args):
    config = vars(args)

    logger.info("Starting synthetic data generation...")
    logger.info(f"Using ray: {config['use_ray']}")
    logger.info(f"Generating {config['N']} pairs of synthetic data.")
    logger.info("Mode: {}".format(config["mode"]))

    start_time = time.time()
    make_synthetic_dataset(config)
    logger.info("Synthetic data generation complete.")
    logger.info(
        f"With use_ray set to {config['use_ray']}, total time: {time.time() - start_time} seconds."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SBRNet with command-line parameters."
    )
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train","test"] ,help="train or test mode. test is with ls"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="output directory to save stack and rfv measurements",
    )
    parser.add_argument(
        "--lower_sbr",
        type=float,
        help="lower bound for SBR",
    )
    parser.add_argument(
        "--upper_sbr",
        type=float,
        help="upper bound for SBR",
    )
    parser.add_argument(
        "--scattering_length", type=float, help="scattering length in microns"
    )
    parser.add_argument(
        "--z_sampling", type=float, default=25, help="z discretization in microns"
    )
    parser.add_argument(
        "--N",
        type=int,
        help="number of pairs to generate",
    )
    parser.add_argument(
        "--use_ray",
        type=bool,
        default=True,
        help="whether to use ray for parallelization",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default="/ad/eng/research/eng_research_cisl/jalido/sbrnet/data/synthetic_vasculature/gt_vasculature/",
        help="folder of ground truths",
    )
    parser.add_argument(
        "--psf_path",
        type=str,
        default="/ad/eng/research/eng_research_cisl/jalido/sbrnet/data/cm2v2/cm2v2psf.tif",
        help="path of psf",
    )
    parser.add_argument(
        "--lenslet_apodize_path",
        type=str,
        default="/ad/eng/research/eng_research_cisl/jalido/sbrnet/data/cm2v2/lensletapodize.tif",
        help="path of lenslet apodization function",
    )
    parser.add_argument(
        "--mla_apodize_path",
        type=str,
        default="/ad/eng/research/eng_research_cisl/jalido/sbrnet/data/cm2v2/mlaapodize.tif",
        help="path of mla apodization function",
    )
    parser.add_argument(
        "--value_path",
        type=str,
        default="/ad/eng/research/eng_research_cisl/jalido/sbrnet/data/valuenoise/",
        help="folder of value noise samples",
    )
    parser.add_argument(
        "--view_ind",
        type=int,
        help="view combination index for object in constants.py",
        required=True,
    )

    # Parse the arguments
    args = parser.parse_args()

    main(args)
