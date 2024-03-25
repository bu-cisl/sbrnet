# Author: Anastasios Nikolas Angelopoulos, angelopoulos@berkeley.edu
# https://github.com/aangelopoulos/im2im-uq/blob/main/core/models/losses/pinball.py
# pinball loss class
import torch
import torch.nn as nn
import logging
from torch.nn.functional import relu

logger = logging.getLogger(__name__)


# pinball loss class
class PinballLoss:
    def __init__(self, quantile=0.10, reduction="mean"):
        self.quantile = quantile
        assert 0 < self.quantile
        assert self.quantile < 1
        self.reduction = reduction
        logger.info(f"Initialized PinballLoss with quantile {self.quantile}")

    def __call__(self, output, target):
        assert output.shape == target.shape
        loss = torch.zeros_like(target, dtype=torch.float)
        error = output - target
        smaller_index = error < 0
        bigger_index = 0 < error
        loss[smaller_index] = self.quantile * (abs(error)[smaller_index])
        loss[bigger_index] = (1 - self.quantile) * (abs(error)[bigger_index])

        if self.reduction == "sum":
            loss = loss.sum()
        if self.reduction == "mean":
            loss = loss.mean()

        return loss


class QuantileLoss(nn.Module):
    def __init__(self, params):
        super(QuantileLoss, self).__init__()

        self.q_lo_loss = PinballLoss(quantile=params["q_lo"])
        self.q_hi_loss = PinballLoss(quantile=params["q_hi"])
        self.point_weight = params["point_loss_weight"]
        self.qlo_weight = params["qlo_weight"]
        self.qlo_pinball_weight = params["qlo_pinball_weight"]
        self.qlo_bce_weight = params["qlo_bce_weight"] 
        self.qhi_weight = params["qhi_weight"]
        self.output_activation_name = params["output_activation"]

        # Initialize the point loss criterion based on the configuration
        if params["criterion_name"] == "bce_with_logits":
            self.point_loss = nn.BCEWithLogitsLoss()
        elif params["criterion_name"] == "mse":
            self.point_loss = nn.MSELoss()
        elif params["criterion_name"] == "mae":
            self.point_loss = nn.L1Loss()
        else:
            print(
                f"Unknown loss criterion: {params['criterion_name']}. Using BCEWithLogitsLoss."
            )

        #self.slice_q_lo = slice(0, params["num_gt_layers"]) #add bce weight here
        self.slice_q_lo_pinball = slice(0, params["num_gt_layers"])  #slice for qlo pinball
        self.slice_q_lo_bce = slice(params["num_gt_layers"], params["num_gt_layers"] * 2) #slice for qlo bce
        self.slice_q_hi = slice(params["num_gt_layers"]*2, params["num_gt_layers"] * 3)
        self.slice_point = slice(params["num_gt_layers"] * 3, None)

    def output_activation(self, x):
        if self.output_activation_name == "sigmoid":
            return torch.sigmoid(x)
        elif self.output_activation_name == "relu":
            return relu(x, inplace=False)
        elif self.output_activation_name == "none":
            return x
        else:
            logger.info(
                f"Unknown output activation: {self.output_activation_name}. Using sigmoid."
            )
            return torch.sigmoid(x)

    def forward(self, pred, target):
        '''
        qlo_loss = self.qlo_weight * self.q_lo_loss(
            self.output_activation(pred[:, self.slice_q_lo, :, :]), target
        )
        '''
        qlo_pinball_loss = self.qlo_pinball_weight * self.q_lo_loss(
            self.output_activation(pred[:, self.slice_q_lo_pinball, :, :]), target
        )
        qlo_bce_loss = self.qlo_bce_weight * self.point_loss(
            pred[:, self.slice_q_lo_bce, :, :], target
        )
        qlo_loss = self.qlo_weight * (qlo_pinball_loss + qlo_bce_loss)
        
        point_pred_loss = self.point_weight * self.point_loss(
            pred[:, self.slice_point, :, :], target
        )
        qhi_loss = self.qhi_weight * self.q_hi_loss(
            self.output_activation(pred[:, self.slice_q_hi, :, :]), target
        )
        loss = qlo_loss + qhi_loss + point_pred_loss

        return loss, qlo_loss, qhi_loss, point_pred_loss
