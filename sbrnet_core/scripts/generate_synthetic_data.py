import os
import random
import pandas as pd
from utils import utils, sbrnet_utils
import time

low_sbr = 1.1
high_sbr = 3.0

# TODO: need to make config file for all file paths so they're not hard-coded
psf_path = "/ad/eng/research/eng_research_cisl/jalido/sbrnet/data/cm2v2/z_uninterpolated_PSF.tif"
PSF = utils.normalize_psf_power(utils.full_read(psf_path))
gt_folder = (
    "/ad/eng/research/eng_research_cisl/jalido/sbrnet/data/synthetic_vasculature/beads/"
)

lens_apodize_path = (
    "/ad/eng/research/eng_research_cisl/jalido/sbrnet/data/cm2v2/lensletapodize.tiff"
)
LENS_AP = utils.full_read(lens_apodize_path)
mla_apodize_path = (
    "/ad/eng/research/eng_research_cisl/jalido/sbrnet/data/cm2v2/mlaapodize.tiff"
)
MLA_AP = utils.full_read(mla_apodize_path)

# for nicholas to change to his own folder in projectnb
out_folder = "/ad/eng/research/eng_research_cisl/"
out_stack_folder = os.path.join(out_folder, "stack")
out_rfv_folder = os.path.join(out_folder, "rfv")
if not os.path.exists(out_folder):
    # If the folder doesn't exist, create it
    os.makedirs(out_folder)
    os.makedirs(out_stack_folder)
    os.makedirs(out_rfv_folder)

df = pd.DataFrame()
t0 = time.time()
for i in range(500):
    sbr = random.uniform(low_sbr, high_sbr)
    gt_path = os.path.join(gt_folder, f"gt_vol_{i}.tiff")
    gt = utils.full_read(gt_path)

    value_path = f"/ad/eng/research/eng_research_cisl/jalido/sbrnet/data/valuenoise/value_{i+1}.png"
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
    df = pd.concat([df, rowdata], ignore_index=True, axis=0)

df.to_parquet(out_folder + "metadata.pq")
