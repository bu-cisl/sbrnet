import logging
import argparse
from datetime import datetime
from torch import compile

from sbrnet_core.sbrnet.model import SBRNet

# from sbrnet_core.config_loader import load_config # Not needed anymore
from sbrnet_core.sbrnet.trainer import Trainer

# Get the current timestamp as a string
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define the log file path with the timestamp
log_file_path = f"/projectnb/tianlabdl/jalido/sbrnet_proj/.log/logging/sbrnet_train_{current_time}.log"

# Configure logging to write log messages to the file
logging.basicConfig(filename=log_file_path, level=logging.INFO)

logger = logging.getLogger(__name__)


def main(args):
    # Construct the configuration dictionary from the argparse namespace
    config = vars(args)

    model = compile(SBRNet(config))

    trainer = Trainer(model, config)

    logger.info("Starting training...")

    trainer.train()

    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SBRNet with command-line parameters."
    )

    # paths
    parser.add_argument(
        "--dataset_pq",
        type=str,
        required=True,
        help="Path to the Parquet dataset file.",
    )
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Directory to save trained models."
    )

    # training stuff
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="The ratio of training set split.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--epochs", type=int, default=20000, help="Number of epochs for training."
    )
    parser.add_argument(
        "--backbone", type=str, default="resnet", help="The backbone network type."
    )
    parser.add_argument(
        "--resnet_channels",
        type=int,
        default=48,
        help="Number of channels in resnet backbone.",
    )
    parser.add_argument(
        "--weight_init",
        type=str,
        default="kaiming_normal",
        help="Weight initialization method.",
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Seed for random number generators."
    )
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer type.")
    parser.add_argument(
        "--criterion_name",
        type=str,
        default="bce_with_logits",
        help="Criterion name for the loss function.",
    )
    parser.add_argument(
        "--use_amp",
        type=bool,
        default=True,
        help="Whether to use automatic mixed precision or not.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine_annealing",
        help="Learning rate scheduler type.",
    )
    parser.add_argument(
        "--cosine_annealing_T_max",
        type=int,
        default=30,
        help="Maximum number of iterations for cosine annealing scheduler.",
    )

    # model stuff
    parser.add_argument(
        "--num_gt_layers", type=int, default=24, help="Number of ground truth layers."
    )
    parser.add_argument(
        "--num_lf_views", type=int, default=9, help="Number of light field views."
    )
    parser.add_argument(
        "--num_rfv_layers", type=int, default=24, help="Number of RFV layers."
    )
    parser.add_argument(
        "--num_resblocks", type=int, default=20, help="Number of residual blocks."
    )
    parser.add_argument(
        "--patch_size", type=int, default=224, help="Size of the patch."
    )

    # calibrated parameters for poisson-gaussian noise model
    parser.add_argument(
        "--A_STD",
        type=float,
        default=5.7092e-5,
        help="Standard deviation of A for poisson-gaussian noise model.",
    )
    parser.add_argument(
        "--A_MEAN",
        type=float,
        default=1.49e-4,
        help="Mean of A for poisson-gaussian noise model.",
    )
    parser.add_argument(
        "--B_STD",
        type=float,
        default=2.7754e-6,
        help="Standard deviation of B for poisson-gaussian noise model.",
    )
    parser.add_argument(
        "--B_MEAN",
        type=float,
        default=5.41e-6,
        help="Mean of B for poisson-gaussian noise model.",
    )

    # Parse the arguments
    args = parser.parse_args()

    main(args)
