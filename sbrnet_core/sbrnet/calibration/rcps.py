from typing import Tuple
from venv import logger
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
from torch.nn.functional import relu

# import os
# import multiprocessing
import logging

logger = logging.getLogger(__name__)


def algorithm(
    cal_data: Dataset,
    model: Module,
    alpha: float,
    delta: float,
    lamb_0: torch.Tensor,
    d_lamb: float,
    params: dict,
) -> torch.Tensor:
    lamb = lamb_0
    UCB = torch.zeros(len(lamb), device="cuda")
    while torch.any(UCB < alpha):
        lamb -= d_lamb * (UCB <= alpha)
        L = calc_all_L(cal_data=cal_data, model=model, lamb=lamb, params=params)
        UCB = calc_UCB(L=L, delta=delta, N=len(cal_data))
        print("lambdas: ", lamb)
        print("UCB: ", UCB)
        print("------------------")
    return lamb + d_lamb


def calc_UCB(L: torch.Tensor, delta: float, N: int) -> torch.Tensor:
    """upper confidence bound hoeffding

    Args:
        L (torch.Tensor): size (num_batches, num_gt_layers)
        delta (torch.float): float between 0 and 1
        N: number of samples

    Returns:
        torch.Tensor: size of num_gt_layers
    """
    return torch.mean(L, dim=0) + torch.sqrt(
        1 / (2 * N) * torch.log(1 / torch.tensor(delta))
    )


def calc_all_L(cal_data: Dataset, model, lamb, params: dict) -> torch.Tensor:
    """table of Ls. includes size of dataset in the summary statistics since i divide by N in the L calculation

    Args:
        cal_data (Dataset): _description_
        model (_type_): _description_
        lamb (_type_): _description_
        params (dict): _description_

    Returns:
        torch.Tensor: size (num_batches, num_gt_layers)
    """
    dataloader = DataLoader(cal_data, batch_size=12, shuffle=False, pin_memory=True)
    L_table = torch.empty(len(dataloader), params["num_gt_layers"]).cuda()
    for i, (stack, rfv, gt) in enumerate(dataloader):
        stack = stack.cuda()
        rfv = rfv.cuda()
        gt = gt.cuda()
        with torch.no_grad():
            L_table[i, :] = calc_one_L(
                stack=stack, rfv=rfv, gt=gt, model=model, lamb=lamb, params=params
            )
    return L_table


def calc_one_L(
    stack: torch.Tensor,
    rfv: torch.Tensor,
    gt: torch.Tensor,
    model: Module,
    lamb: torch.Tensor,
    params: dict,
) -> torch.Tensor:
    """L_i <- L(T_lamb(X_i))

    Args:
        stack (torch.Tensor): B x num_views x H x W
        rfv (torch.Tensor): B x num_rfv_layers x H x W
        gt (torch.Tensor): B x num_gt_layers x H x W
        model (Module): trained model
        lamb (torch.Tensor): size is num_gt_layers
        params (dict): config dict

    Returns:
        torch.Tensor: size of num_gt_layers
    """
    q_lo, q_hi, f = get_preds(model=model, stack=stack, rfv=rfv, params=params)
    # q_lo, q_hi, f = q_lo/q_hi.max(), q_hi/q_hi.max(), f/q_hi.max()
    low_T_lambda, high_T_lambda = form_T_lambda(f, q_lo, q_hi, lamb)

    num_gt_layers = gt.size(1)
    L_values = []

    for layer_idx in range(num_gt_layers):
        gt_layer = gt[:, layer_idx, :, :]
        low_T_lambda_layer = low_T_lambda[:, layer_idx, :, :]
        high_T_lambda_layer = high_T_lambda[:, layer_idx, :, :]

        # gt_layer = gt_layer / high_T_lambda_layer.max()

        num_elements_between = torch.sum(
            (gt_layer >= low_T_lambda_layer) & (gt_layer < high_T_lambda_layer)
        )
        total_elements = gt_layer.numel()
        L = 1 - num_elements_between / total_elements
        L_values.append(L)

    return torch.stack(L_values)


def form_T_lambda(
    f: torch.Tensor, q_lo: torch.Tensor, q_hi: torch.Tensor, lamb: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        f (torch.Tensor): B x num_gt_layers x H x W
        q_lo (torch.Tensor): B x num_gt_layers x H x W
        q_hi (torch.Tensor): B x num_gt_layers x H x W
        lamb (torch.Tensor): num_gt_layers
    Returns:
        Tuple(torch.Tensor, torch.Tensor): each of shape (batch_size, num_gt_layers, H, W)
    """

    lamb_broadcasted = lamb.view(1, -1, 1, 1)  # 1 x num_gt_layers x 1 x 1
    low_T = f - lamb_broadcasted * (f - q_lo)  # B x num_gt_layers x H x W
    high_T = f + lamb_broadcasted * (q_hi - f)  # B x num_gt_layers x H x W

    return low_T, high_T


def get_preds(
    model: Module, stack: torch.Tensor, rfv: torch.Tensor, params: dict
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """_summary_

    Args:
        model (_type_): model
        stack (torch.Tensor): B x num_views x H x W
        rfv (torch.Tensor): B x num_rfv_layers x H x W
        params (dict): config dict

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: each of shape (batch_size, num_gt_layers, H, W)
    """
    model.eval()
    slice_q_lo = slice(0, params["num_gt_layers"])
    slice_q_hi = slice(params["num_gt_layers"], params["num_gt_layers"] * 2)
    slice_point = slice(params["num_gt_layers"] * 2, None)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            out = model(stack, rfv)
            q_lo = output_activation(out[:, slice_q_lo, :, :], params)
            q_hi = output_activation(out[:, slice_q_hi, :, :], params)
            f = torch.sigmoid(out[:, slice_point, :, :])
    return q_lo, q_hi, f


def output_activation(x, config):
    if config.get("output_activation") == "sigmoid":
        return torch.sigmoid(x)
    elif config.get("output_activation") == "relu":
        return relu(x, inplace=False)
    elif config.get("output_activation") == "none":
        return x
    else:
        logger.info(
            f"Unknown output activation: {config.get('output_activation')}. Using sigmoid."
        )
        return torch.sigmoid(x)


# if __name__ == "__main__":
#     params = None
#     labdas = algorithm(cal_data=cal_data, model=model, alpha=0.1, delta=0.1, lamb_0=lamb_0, d_lamb=0.1, params=params)
