import os
import random
import pandas as pd
from utils import utils, sbrnet_utils
import multiprocessing
from functools import partial
import time


def process_iteration(
    i,
    low_sbr,
    high_sbr,
    gt_folder,
    PSF,
    LENS_AP,
    MLA_AP,
    out_stack_folder,
    out_rfv_folder,
    out_folder,
    shared_df,
):
    sbr = random.uniform(low_sbr, high_sbr)
    gt_path = os.path.join(gt_folder, f"gt_vol_{i}.tiff")
    gt = utils.full_read(gt_path)

    value_path = f"/projectnb/tianlabdl/jalido/sbrnet/data/valuenoise/value_{i+1}.png"
    value = utils.full_read(value_path)

    fs_meas = utils.lsi_fwd_mdl(utils.pad_3d_array_to_size(gt, PSF.shape), PSF)
    print("here")
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
            "psf_path": [psf_path],
            "lens_apodized_path": [lens_apodize_path],
            "mla_apodized_path": [mla_apodize_path],
            "gt_folder": [gt_folder],
            "value_path": [value_path],
            "sbr": [sbr],
            "stack_path": [out_stack_path],
            "rfv_path": [out_rfv_path],
        }
    )

    with shared_df.get_lock():
        shared_df.value = pd.concat(
            [shared_df.value, rowdata], ignore_index=True, axis=0
        )


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
    shared_df = manager.Value(pd.DataFrame)

    pool = multiprocessing.Pool(processes=num_processes)
    t0 = time.time()
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
        out_folder=out_folder,
        shared_df=shared_df,
    )
    print("multiprocessing", time.time(-t0))

    pool.map(partial_func, range(num_iterations))
    pool.close()
    pool.join()

    final_df = shared_df.value
    final_df.to_parquet(os.path.join(out_folder, "metadata.pq"))
