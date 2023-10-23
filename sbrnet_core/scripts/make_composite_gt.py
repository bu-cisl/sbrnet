from utils.utils import *

vasc_path = "/projectnb/tianlabdl/jalido/synthetic_vasculature/16um_diameter/"
bead_path = "/projectnb/tianlabdl/jalido/synthetic_vasculature/beads/"
out_path = "/projectnb/tianlabdl/jalido/sbrnet/data/datap5vasc/"

for i in range(500):
    beads = linear_normalize(read_tiff(bead_path + "sim_gt_vol_" + str(i) + ".tif"))
    vasc = linear_normalize(read_tiff(vasc_path + "Lnet_i_" + str(i) + ".tiff"))
    gt = linear_normalize(beads + 0.5 * vasc)
    write_tiff(gt, out_path + f"gt_vol_{i}.tiff")
