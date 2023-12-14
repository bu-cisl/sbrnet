import ray
import os
import random
import pandas as pd
import logging
from sbrnet_core.synthetic_data.constants import NUM_SLICES, view_combos
from sbrnet_core.utils import (
    normalize_psf_power,
    full_read_tiff,
    full_read,
    lsi_fwd_mdl,
    pad_3d_array_to_size,
    write_tiff16,
    sbrnet_utils,
)


logger = logging.getLogger(__name__)


@ray.remote
def process_single_iteration(i, config):
    view_combo_index = config["view_ind"] - 1
    view_list = view_combos[view_combo_index]

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

    sbr = random.uniform(low_sbr, upper_sbr)
    gt_path = os.path.join(gt_folder, f"sim_gt_vol_{i}.tif")
    gt = full_read_tiff(gt_path)

    value_path = os.path.join(value_folder, f"value_{i+1}.png")
    value = full_read(value_path)

    # make free space image
    fs_meas = lsi_fwd_mdl(pad_3d_array_to_size(gt, PSF.shape), PSF)

    # make background image
    bg_meas, bg_mean = sbrnet_utils.make_bg_img(value, LENS_AP, MLA_AP)

    # make synthetic measurement
    synthetic_scat_measurement = sbrnet_utils.make_measurement(
        fs_meas, bg_meas, sbr, bg_mean
    )

    stack_scat = sbrnet_utils.crop_views(im=synthetic_scat_measurement)

    # use only the views in the view list
    stack_scat = sbrnet_utils.zero_slices_not_in_list(stack_scat, view_list)

    rfv_scat = sbrnet_utils.lf_refocus_volume(
        lf=stack_scat, z_slices=NUM_SLICES, max_shift=NUM_SLICES // 2 + 1
    )

    # now get rid of the views that are not in the view list after refocusing.
    # refocusing needs 9 x h x w array to work, so we just zero out the views that are not in the view list
    stack_scat = stack_scat[view_list, :, :]

    #### for free space #####
    stack_free = sbrnet_utils.crop_views(im=fs_meas)

    # use only the views in the view list
    stack_free = sbrnet_utils.zero_slices_not_in_list(stack_free, view_list)

    rfv_free = sbrnet_utils.lf_refocus_volume(
        lf=stack_free, z_slices=NUM_SLICES, max_shift=NUM_SLICES // 2 + 1
    )

    # now get rid of the views that are not in the view list after refocusing.
    # refocusing needs 9 x h x w array to work, so we just zero out the views that are not in the view list
    stack_free = stack_free[view_list, :, :]
    #### for free space #####

    # write to folders
    out_folder = config["out_dir"]
    out_stack_scat_folder = os.path.join(out_folder, "stack_scattering")
    out_rfv_scat_folder = os.path.join(out_folder, "rfv_scattering")
    out_stack_free_folder = os.path.join(out_folder, "stack_freespace")
    out_rfv_free_folder = os.path.join(out_folder, "rfv_freespace")

    out_stack_scat_path = os.path.join(out_stack_scat_folder, f"meas_{i}.tiff")
    out_stack_free_path = os.path.join(out_stack_free_folder, f"meas_{i}.tiff")
    out_rfv_scat_path = os.path.join(out_rfv_scat_folder, f"meas_{i}.tiff")
    out_rfv_free_path = os.path.join(out_rfv_free_folder, f"meas_{i}.tiff")

    write_tiff16(stack_scat, out_stack_scat_path)
    write_tiff16(rfv_scat, out_rfv_scat_path)
    write_tiff16(stack_free, out_stack_free_path)
    write_tiff16(rfv_free, out_rfv_free_path)

    rowdata = pd.DataFrame(
        {
            "num_views": [len(view_list)],
            "view_combo": [view_list],
            "psf_path": [psf_path],
            "lens_apodized_path": [lens_apodize_path],
            "mla_apodized_path": [mla_apodize_path],
            "gt_folder": [gt_folder],
            "value_path": [value_path],
            "sbr": [sbr],
            "sbr_range": [f"{low_sbr}-{upper_sbr}"],
            "stack_scat_path": [out_stack_scat_path],
            "rfv_scat_path": [out_rfv_scat_path],
            "stack_free_path": [out_stack_free_path],
            "rfv_free_path": [out_rfv_free_path],
            "gt_path": [gt_path],
        }
    )
    logger.info(f"Finished iteration {i}")
    return rowdata


def make_synthetic_dataset(config: dict) -> None:
    out_folder = config["out_dir"]
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    out_stack_folder = os.path.join(out_folder, "stack_scattering")
    if not os.path.exists(out_stack_folder):
        os.makedirs(out_stack_folder)
    out_rfv_folder = os.path.join(out_folder, "rfv_scattering")
    if not os.path.exists(out_rfv_folder):
        os.makedirs(out_rfv_folder)
    out_stack_folder = os.path.join(out_folder, "stack_freespace")
    if not os.path.exists(out_stack_folder):
        os.makedirs(out_stack_folder)
    out_rfv_folder = os.path.join(out_folder, "rfv_freespace")
    if not os.path.exists(out_rfv_folder):
        os.makedirs(out_rfv_folder)

    if config["use_ray"]:
        # num_cpus for SCC--number of cores in omp instance
        ray.init(ignore_reinit_error=True, num_cpus=int(os.getenv("NSLOTS")))
        futures = [
            process_single_iteration.remote(i, config) for i in range(config["N"])
        ]
        results = ray.get(futures)
        df = pd.concat(results, ignore_index=True, axis=0)
    # BUG: This is not working. The code below is not being executed
    else:
        df = pd.DataFrame()
        for i in range(config["N"]):
            rowdata = process_single_iteration(i, config)  # Call the function directly
            df = pd.concat([df, rowdata], ignore_index=True, axis=0)

    df.to_parquet(
        os.path.join(out_folder, "metadata.pq")
    )  # save metadata about dataset in a parquet file
