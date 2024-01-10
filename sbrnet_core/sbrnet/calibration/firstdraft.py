from sbrnet_core.sbrnet.calibration.rcps import algorithm
from sbrnet_core.sbrnet.dataset import CustomDataset
from sbrnet_core.sbrnet.models.model import SBRNet
import torch
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
view_ind = 6
config = {
    "dataset_pq": f"/ad/eng/research/eng_research_cisl/jalido/sbrnet/data/training_data/UQ/{view_ind}/test_metadata.pq",
    "scattering": "scat",
    "train_split": 0.8,
    "batch_size": 16,
    "learning_rate": 0.001,
    "epochs": 20000,
    "backbone": "resnet",
    "resnet_channels": 48,
    "weight_init": "kaiming_normal",
    "random_seed": 42,
    "optimizer": "adam",
    "criterion_name": "bce_with_logits",
    "use_amp": True,
    "lr_scheduler": "cosine_annealing_with_warm_restarts",
    "weight_decay": 0.001,
    "cosine_annealing_T_max": 100,
    "q_lo": 0.05,
    "q_hi": 0.95,
    "num_gt_layers": 24,
    "num_lf_views": 1,
    "num_rfv_layers": 24,
    "num_resblocks": 15,
    "patch_size": 224,
    "use_quantile_layer": True,
    "num_head_layers": 3,
    "view_ind": view_ind,
    "batch_size": 20
}

cal_data = CustomDataset(config)
model = SBRNet(config).cuda()
checkpoint = torch.load(f'/projectnb/tianlabdl/jalido/sbrnet_proj/trained_models/sbrnet_view_{view_ind}_v1.pt',map_location=torch.device('cpu'))
mod_state_dict = checkpoint['model_state_dict']
for key in list(mod_state_dict.keys()):
    mod_state_dict[key.replace("_orig_mod.", "")] = mod_state_dict.pop(key)
model.load_state_dict(mod_state_dict)
lambdas = algorithm(cal_data=cal_data, model=model, alpha=0.1, delta=0.1, lamb_0=2*torch.ones(24).cuda(), d_lamb=0.05, params=config)
print(lambdas)