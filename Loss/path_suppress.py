import torch
import torch.nn.functional as F
from torch import Tensor as Tensor
from torch import tensor as tensor
from torch.nn import CTCLoss
from torch.nn import LogSoftmax
from torch.utils.data import WeightedRandomSampler
# import numpy as np

finfo_min_fp32 = torch.finfo(torch.float32).min
finfo_min_fp16 = torch.finfo(torch.float16).min

# @ torch.jit.script
# def logadd(*args):
#     return torch.logsumexp(torch.stack(args), dim=0)

@ torch.jit.script
def logadd(arg1, arg2, arg3):
    return torch.logsumexp(torch.stack([arg1, arg2, arg3]), dim=0)





def forward(log_probs_ro: Tensor, # (spatial_len, time_len, B, 9)
            max_paths: int, #
            direction: str,
            zero: Tensor,
            two_directions: bool
            ):
    if direction == 'hori':
        log_probs_ro_ = [logadd(log_probs_ro[:, :, :, 0], log_probs_ro[:, :, :, 3], log_probs_ro[:, :, :, 6])]
        if two_directions:
            log_probs_ro_ += [logadd(log_probs_ro[:, :, :, 1], log_probs_ro[:, :, :, 2], log_probs_ro[:, :, :, 4])]
        else:
            log_probs_ro_ += [log_probs_ro[:, :, :, 4]]

    elif direction == 'vert':
        log_probs_ro_ = [logadd(log_probs_ro[:, :, :, 0], log_probs_ro[:, :, :, 1], log_probs_ro[:, :, :, 2])]
        if two_directions:
            log_probs_ro_ += [logadd(log_probs_ro[:, :, :, 3], log_probs_ro[:, :, :, 2], log_probs_ro[:, :, :, 4])]
        else:
            log_probs_ro_ += [log_probs_ro[:, :, :, 4]]
    log_probs_ro_ = torch.stack(log_probs_ro_, dim=-1) #(spatial_len, time_len, B, 2)
    log_probs_ro_ = log_probs_ro_.permute(0, 2, 1, 3).flatten(start_dim=1, end_dim=2)  # (spatial_len, B*time_len, 9)
    spatial_len, nbbt, _ = log_probs_ro_.shape

    expanded_targets = torch.full((nbbt, max_paths+1), fill_value=0, dtype=torch.int64, device=log_probs_ro_.device)
    expanded_targets = torch.stack([torch.full_like(expanded_targets, 1), expanded_targets], dim=-1).flatten(
        start_dim=-2)
    expanded_targets = expanded_targets[:, :-1]  # (nbb, 2*max_paths+1)

    log_probs_ro__ = log_probs_ro_.gather(-1, expanded_targets.expand(spatial_len, -1, -1)) # (spatial_len, nbbt, 2*max_paths+1)

    padding = 2
    log_alpha = torch.full((spatial_len, nbbt, padding+expanded_targets.shape[-1]), fill_value=zero,
                           dtype=log_probs_ro.dtype, device=log_probs_ro.device) # (spatial_len, nbbt, padding + 2*max_paths+1)
    log_alpha[0, :, padding+0] = log_probs_ro__[0, :, 0]
    log_alpha[0, :, padding+1] = log_probs_ro__[0, :, 1]
    alpha = log_alpha.permute(1, 2, 0).detach().cpu().numpy()
    is_blank = expanded_targets == 1
    for t in range(1, spatial_len):
        log_alpha[t, :, 2:] = log_probs_ro__[t, :, :] + logadd(torch.where(is_blank, log_alpha[t-1, :, 2:], zero),
                                                               log_alpha[t-1, :, 1:-1],
                                                               log_alpha[t-1, :, :-2])

    return log_alpha[:, :, padding:]

def path_suppress_ctc(log_probs_ro: Tensor, # (H, W, B, 9)
                      input_heights: list, # B
                      input_widths: list, # B
                      nb_paths: list, # [] * B
                      reduction: str,
                      trunc_width: bool,
                      trunc_height: bool,
                      directions: list, #
                      ):
    H, W, batch_size, _ = log_probs_ro.shape
    B = torch.arange(batch_size, device=log_probs_ro.device)
    zero = tensor(finfo_min_fp16 if log_probs_ro.dtype == torch.float16 else finfo_min_fp32,
                  device=log_probs_ro.device, dtype=log_probs_ro.dtype)
    max_paths = max(nb_paths)
    l1l2 = []
    for direction in directions:
        assert direction in ['hori', 'vert']
        if direction == 'hori':
            log_alpha = forward(log_probs_ro, max_paths, direction, zero, len(directions) == 2) # (spatial_len, B*time_len, 2*max_paths+1)
            time_len = input_widths if trunc_width else [log_probs_ro.shape[1]] * batch_size
            spatial_len = input_heights if trunc_height else [log_probs_ro.shape[0]] * batch_size
            max_time_len = log_probs_ro.shape[1]
        elif direction == 'vert':
            log_alpha = forward(log_probs_ro.permute(1, 0, 2, 3), max_paths, direction, zero, len(directions) == 2)
            time_len = input_heights if trunc_height else [log_probs_ro.shape[0]] * batch_size
            spatial_len = input_widths if trunc_width else [log_probs_ro.shape[1]] * batch_size

        # alpha = log_alpha[:, 0, :].permute(1, 0).detach().cpu().numpy()
        l1l2_ = []
        for b in range(batch_size):
            log_alpha_ = log_alpha[:spatial_len[b], b*max_time_len:b*max_time_len+time_len[b], nb_paths[b]*2-1: nb_paths[b]*2+1]
            l1l2_.append(torch.sum(torch.logsumexp(log_alpha_[-1], dim=-1)))
        l1l2_ = torch.stack(l1l2_)
        l1l2.append(l1l2_)


    loss = - torch.sum(torch.stack(l1l2, dim=0), dim=0)

    if reduction.lower() == 'sum':
        loss = torch.sum(loss)
    elif reduction.lower() == 'mean':
        loss = torch.mean(loss)

    if loss.isinf():
        print('debug')

    return loss