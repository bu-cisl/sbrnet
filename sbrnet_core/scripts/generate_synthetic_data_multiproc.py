import os
import random
import numpy as np
import pandas as pd
from utils import utils, sbrnet_utils
import multiprocessing
from functools import partial
import time


def process_iteration(
    i: int,
    low_sbr: np.float32,
    high_sbr: np.float32,
    gt_folder: str,
    PSF: np.ndarray,
    LENS_AP: np.ndarray,
    MLA_AP: np.ndarray,
    out_stack_folder: str,
    out_rfv_folder: str,
) -> pd.DataFrame:
    """iteration to generate a single stack/rfv pair from a GT object.

    Args:
        i (int): iteration number
        low_sbr (np.float32): low SBR bound
        high_sbr (np.float32): high SBR bound
        gt_folder (str): where the GT volume is stored
        PSF (np.ndarray): the PSF stack
        LENS_AP (np.ndarray): the lenslet apodization function
        MLA_AP (np.ndarray): the whole MLA apodization function
        out_stack_folder (str): the folder to save the stack of CM2 views
        out_rfv_folder (str): the folder to save the RFV of the CM2 LightField measurement

    Returns:
        pd.DataFrame: DataFrame containing metadata about the training set
    """
    sbr = random.uniform(low_sbr, high_sbr)
    print(i, sbr)
    gt_path = os.path.join(gt_folder, f"gt_vol_{i}.tiff")
    gt = utils.full_read(gt_path)

    value_path = f"/projectnb/tianlabdl/jalido/sbrnet/data/valuenoise/value_{i+1}.png"
    value = utils.full_read(value_path)

    fs_meas = utils.lsi_fwd_mdl(utils.pad_3d_array_to_size(gt, PSF.shape), PSF)

    bg_meas, bg_mean = sbrnet_utils.make_bg_img(value, LENS_AP, MLA_AP)

    synthetic_measurement = sbrnet_utils.make_measurement(
        fs_meas, bg_meas, sbr, bg_mean
    )
    stack = sbrnet_utils.crop_views(synthetic_measurement)
    rfv = sbrnet_utils.lf_refocus_volume(stack, 24, 13)  # number of z slices

    out_stack_path = os.path.join(out_stack_folder, f"meas_{i}.tiff")
    out_rfv_path = os.path.join(out_rfv_folder, f"meas_{i}.tiff")

    utils.write_tiff(stack, out_stack_path)
    utils.write_tiff(rfv, out_rfv_path)

    rowdata = pd.DataFrame(
        {
            "psf_path": [
                psf_path
            ],  # You might need to define psf_path and other variables as appropriate
            "lens_apodized_path": [lens_apodize_path],
            "mla_apodized_path": [mla_apodize_path],
            "gt_folder": [gt_folder],
            "value_path": [value_path],
            "sbr": [sbr],
            "stack_path": [out_stack_path],
            "rfv_path": [out_rfv_path],
        }
    )

    return rowdata


if __name__ == "__main__":
    low_sbr = 1.1
    high_sbr = 3.0
    psf_path = "/projectnb/tianlabdl/jalido/sbrnet/data/cm2v2/z_uninterpolated_PSF.tif"
    PSF = utils.normalize_psf_power(utils.full_read(psf_path))
    gt_folder = "/projectnb/tianlabdl/jalido/sbrnet/data/datap5vasc/"
    lens_apodize_path = (
        "/projectnb/tianlabdl/jalido/sbrnet/data/cm2v2/lensletapodize.tiff"
    )
    LENS_AP = utils.full_read(lens_apodize_path)
    mla_apodize_path = "/projectnb/tianlabdl/jalido/sbrnet/data/cm2v2/mlaapodize.tiff"
    MLA_AP = utils.full_read(mla_apodize_path)
    out_folder = "/projectnb/tianlabdl/jalido/sbrnet/data/training_data/dataset0/"
    out_stack_folder = os.path.join(out_folder, "stack")
    out_rfv_folder = os.path.join(out_folder, "rfv")

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        os.makedirs(out_stack_folder)
        os.makedirs(out_rfv_folder)

    num_iterations = 500
    num_processes = multiprocessing.cpu_count()

    manager = multiprocessing.Manager()

    # Create a list to collect individual DataFrames from processes
    results = []

    pool = multiprocessing.Pool(processes=num_processes)

    partial_func = partial(
        process_iteration,
        low_sbr=low_sbr,
        high_sbr=high_sbr,
        gt_folder=gt_folder,
        PSF=PSF,
        LENS_AP=LENS_AP,
        MLA_AP=MLA_AP,
        out_stack_folder=out_stack_folder,
        out_rfv_folder=out_rfv_folder,
    )

    t0 = time.time()
    # Map the partial_func to the range of num_iterations
    for i in range(num_iterations):
        rowdata = pool.apply_async(
            partial_func, args=(i,)
        ).get()  # Get the returned DataFrame
        results.append(rowdata)  # Append the DataFrame to results
    print("multiprocessing", time.time() - t0)
    # Close and join the pool
    pool.close()
    pool.join()

    # Merge individual DataFrames into the final DataFrame
    final_df = pd.concat(results, ignore_index=True, axis=0)
    final_df.to_parquet(os.path.join(out_folder, "metadata.pq"))
