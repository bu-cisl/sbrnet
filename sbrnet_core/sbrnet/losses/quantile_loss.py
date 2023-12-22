# Author: Anastasios Nikolas Angelopoulos, angelopoulos@berkeley.edu
# https://github.com/aangelopoulos/im2im-uq/blob/main/core/models/losses/pinball.py
# pinball loss class
import torch
import torch.nn as nn


# pinball loss class
class PinballLoss:
    def __init__(self, quantile=0.10, reduction="mean"):
        self.quantile = quantile
        assert 0 < self.quantile
        assert self.quantile < 1
        self.reduction = reduction

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

        self.slice_q_lo = slice(0, params["num_gt_layers"])
        self.slice_q_hi = slice(params["num_gt_layers"], params["num_gt_layers"] * 2)
        self.slice_point = slice(params["num_gt_layers"] * 2, None)

    def forward(self, pred, target):
        loss = (
            self.q_lo_loss(pred[:, self.slice_q_lo, :, :], target)
            + self.q_hi_loss(pred[:, self.slice_q_hi, :, :], target)
            + self.point_loss(pred[:, self.slice_point, :, :], target)
        )

        return loss