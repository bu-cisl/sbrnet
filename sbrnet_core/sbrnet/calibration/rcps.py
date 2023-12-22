from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module





def calc_UCB(L_list: torch.Tensor, delta: torch.float) -> torch.float:
    """upper confidence bound by hoeffding

    Args:
        L_list (torch.Tensor): List of L_i's
        delta (torch.float): number between 0 and 1.

    Returns:
        torch.float: _description_
    """
    return torch.mean(L_list) + torch.sqrt(1/(2*len(L_list)) * torch.log(1/delta))


def calc_all_L(cal_data: Dataset, model, lamb, params: dict) -> torch.Tensor:
    dataloader = DataLoader(cal_data, batch_size=16, shuffle=False, pin_memory=True)
    L_list = torch.empty(len(dataloader), dtype=torch.float)
    for i, (stack, rfv, gt) in enumerate(dataloader):
        stack = stack.cuda()
        rfv = rfv.cuda()
        gt = gt.cuda()
        with torch.no_grad():
            L_list[i] = calc_one_L(
                stack=stack, rfv=rfv, gt=gt, model=model, lamb=lamb, params=params
            )
    return L_list


def calc_one_L(
    stack: torch.Tensor, rfv: torch.Tensor, gt: torch.Tensor, model, lamb, params: dict
) -> torch.float:
    """L_i <- L(T_lamb(X_i))

    Args:
        stack (torch.Tensor): _description_
        rfv (torch.Tensor): _description_
        gt (torch.Tensor): _description_
        model (_type_): _description_
        lamb (_type_): _description_
        params (dict): _description_

    Returns:
        torch.float: _description_
    """
    q_lo, q_hi, f = get_preds(model=model, stack=stack, rfv=rfv, params=params)
    low_T_lambda, high_T_lambda = form_T_lambda(f, q_lo, q_hi, lamb)

    num_elements_between = torch.sum((gt > low_T_lambda) & (gt < high_T_lambda))
    total_elements = gt.numel()
    L = 1 - num_elements_between / total_elements
    return L


def form_T_lambda(f, q_lo, q_hi, lamb) -> Tuple(torch.Tensor, torch.Tensor):
    low_T = f - lamb * (f - q_lo)
    high_T = f + lamb * (q_hi - f)
    return low_T, high_T


def get_preds(
    model, stack: torch.Tensor, rfv: torch.Tensor, params: dict
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    slice_q_lo = slice(0, params["num_gt_layers"])
    slice_q_hi = slice(params["num_gt_layers"], params["num_gt_layers"] * 2)
    slice_point = slice(params["num_gt_layers"] * 2, None)

    with torch.no_grad():
        out = model(stack, rfv)
        q_lo = (torch.sigmoid(out[:, slice_q_lo, :, :]),)
        q_hi = (torch.sigmoid(out[:, slice_q_hi, :, :]),)
        f = out[:, slice_point, :, :]
    return q_lo, q_hi, f
