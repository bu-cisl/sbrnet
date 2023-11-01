import os
import random
import pandas as pd
import logging
from sbrnet_core.utils import (
    normalize_psf_power,
    full_read_tiff,
    full_read,
    lsi_fwd_mdl,
    pad_3d_array_to_size,
    write_tiff,
    sbrnet_utils,
)

logger = logging.getLogger(__name__)


def make_synthetic_dataset(config: dict) -> None:
    """script to take config to generate synthetic dataset.
    Requires existing datasets of ground truth objects and value noise samples.
    Also requires the system PSF, and apodization functions. These paths are defined in the config.

    Args:
        config (dict): _description_
    """
    low_sbr = config["lower_sbr"]
    upper_sbr = config["upper_sbr"]

    psf_path = config["psf_path"]
    PSF = normalize_psf_power(full_read_tiff(psf_path))
    gt_folder = config["gt_path"]

    value_folder = config["value_path"]

    lens_apodize_path = config["lenslet_apodize_path"]
    LENS_AP = full_read_tiff(lens_apodize_path)
    mla_apodize_path = config["mla_apodize_path"]
    MLA_AP = full_read_tiff(mla_apodize_path)

    out_folder = config["out_path"]
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    out_stack_folder = os.path.join(out_folder, "stack")
    if not os.path.exists(out_stack_folder):
        os.makedirs(out_stack_folder)
    out_rfv_folder = os.path.join(out_folder, "rfv")
    if not os.path.exists(out_rfv_folder):
        os.makedirs(out_rfv_folder)

    df = pd.DataFrame()
    # TODO: parallelize across cores with ray
    for i in range(config["N"]):
        sbr = random.uniform(low_sbr, upper_sbr)
        gt_path = os.path.join(gt_folder, f"gt_vol_{i}.tiff")
        gt = full_read_tiff(gt_path)

        value_path = os.path.join(value_folder, f"value_{i+1}.png")
        value = full_read(value_path)

        fs_meas = lsi_fwd_mdl(pad_3d_array_to_size(gt, PSF.shape), PSF)
        bg_meas, bg_mean = sbrnet_utils.make_bg_img(value, LENS_AP, MLA_AP)

        synthetic_measurement = sbrnet_utils.make_measurement(
            fs_meas, bg_meas, sbr, bg_mean
        )
        stack = sbrnet_utils.crop_views(synthetic_measurement)
        rfv = sbrnet_utils.lf_refocus_volume(
            stack, config["num_slices"], config["num_slices"] // 2 + 1
        )  # number of z slices, max shift

        out_stack_path = os.path.join(out_stack_folder, f"meas_{i}.tiff")
        out_rfv_path = os.path.join(out_rfv_folder, f"meas_{i}.tiff")

        write_tiff(stack, out_stack_path)
        write_tiff(rfv, out_rfv_path)

        rowdata = pd.DataFrame(
            {
                "psf_path": [psf_path],
                "lens_apodized_path": [lens_apodize_path],
                "mla_apodized_path": [mla_apodize_path],
                "gt_folder": [gt_folder],
                "value_path": [value_path],
                "sbr": [sbr],
                "stack_path": [out_stack_path],
                "rfv_path": [out_rfv_path],
                "gt_path": [gt_path],
            }
        )
        df = pd.concat([df, rowdata], ignore_index=True, axis=0)
        logger.info(
            f"Finished synthetic data generation for {i+1} of {config['N']}. SBR: {sbr}"
        )

    df.to_parquet(
        out_folder + "metadata.pq"
    )  # save metadata about dataset in a parquet file
