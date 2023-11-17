import os
from sbrnet_core.utils import full_read_tiff, write_tiff, linear_normalize

# The sbrnet-core package does not have a script to generate the synthetic ground truths.
# This script takes existing ground truths, beads and vasculature, and combines them.
# TODO: migrate synthetic data generation to sbrnet-core from legacy matlab scripts
vasc_path = "/ad/eng/research/eng_research_cisl/jalido/sbrnet/data/synthetic_vasculature/16um_diameter"
bead_path = "/ad/eng/research/eng_research_cisl/jalido/sbrnet/data/beads/"
out_path = "/ad/eng/research/eng_research_cisl/jalido/sbrnet/data/gts/composite/"

if not os.path.exists(out_path):
    os.makedirs(out_path)

for i in range(500):
    beads = full_read_tiff(os.path.join(bead_path, f"sim_gt_vol_{i}.tif"))
    vasc = full_read_tiff(os.path.join(vasc_path, f"Lnet_i_{i}.tiff"))
    gt = linear_normalize(beads + 0.5 * vasc)
    write_tiff(gt, os.path.join(out_path, f"gt_vol_{i}.tiff"))
    print(i)
