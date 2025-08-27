# TODO: try to replace fancy tensor indexing by gather / scatter

import math
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import CTCLoss


finfo_min_fp32 = torch.finfo(torch.float32).min
finfo_min_fp16 = torch.finfo(torch.float16).min


def forward(log_probs: torch.Tensor, # H x W x B x (C+1)
            log_probs_H: torch.Tensor, # H x B x 2
            targets: list,
            target_lengths: list,
            nb_lines: int,
            input_widths: list,
            blank: int,
            lrep: bool = False,
            ):

    input_time_height, input_time_width, batch_size, nc = log_probs.shape
    zero_padding, zero = 2, torch.tensor(finfo_min_fp16 if log_probs.dtype == torch.float16 else finfo_min_fp32,
                                         device=log_probs.device, dtype=log_probs.dtype)

    blank_lines = torch.as_tensor([True, False], device=log_probs.device).expand(batch_size, max(nb_lines)+1, -1).flatten(start_dim=1)
    blank_lines = blank_lines[:, :-1]

    loss_ctc_func = CTCLoss(blank=blank, reduction="none")

    """计算文本行的概率"""
    log_probs_line = torch.full((input_time_height, batch_size, 2*max(nb_lines)+1), fill_value=zero, dtype=log_probs.dtype, device=log_probs.device)
    for u in range(2*max(nb_lines)+1):
        if u % 2 == 0:
            if log_probs_H is not None:
                log_probs_line[:, :, u] = log_probs_H[:, :, 1]
            else:
                log_probs_line[:, :, u] = torch.stack([torch.sum(log_probs[:, :input_widths[i], i, blank], dim=1) for i in range(batch_size)], dim=-1)
        else:
            log_line = - loss_ctc_func(log_probs.permute(1, 0, 2, 3).flatten(start_dim=1, end_dim=2),
                                       targets[u//2].unsqueeze(dim=0).expand(input_time_height, -1, -1).flatten(start_dim=0, end_dim=1),
                                       input_widths * input_time_height,
                                       target_lengths[u//2] * input_time_height)
            log_probs_line[:, :, u] = log_line.unflatten(dim=0, sizes=(input_time_height, batch_size))
            if log_probs_H is not None:
                log_probs_line[:, :, u] += log_probs_H[:, :, 0]


    """计算log_alpha"""
    log_alpha = torch.full((input_time_height, batch_size, zero_padding + 2*max(nb_lines)+1), zero,
                           device=log_probs.device, dtype=log_probs.dtype)
    log_alpha[0, :, zero_padding + 0] = log_probs_line[0, :, 0]
    log_alpha[0, :, zero_padding + 1] = log_probs_line[0, :, 1]

    for t in range(1, input_time_height):
        if not lrep:
            log_alpha[t, :, 2:] = log_probs_line[t] + logadd(torch.where(blank_lines, log_alpha[t - 1, :, 2:], zero),
                                                             log_alpha[t - 1, :, 1:-1],
                                                             torch.where(torch.logical_not(blank_lines), log_alpha[t - 1, :, :-2], zero))
        else:
            log_alpha[t, :, 2:] = log_probs_line[t] + logadd(log_alpha[t - 1, :, 2:],
                                                             log_alpha[t - 1, :, 1:-1],
                                                             torch.where(torch.logical_not(blank_lines), log_alpha[t - 1, :, :-2], zero))
        # log_alpha[t, :, 2:] = torch.where(torch.logical_not(torch.isinf(log_alpha[t, :, 2:])), log_alpha[t, :, 2:], zero)
    return log_alpha[:, :, 2:]



def log_mul(*args, zero=None):
    if not zero:
        zero = torch.tensor(finfo_min_fp16 if args[0].dtype == torch.float16 else finfo_min_fp32,
                            device=args[0].device, dtype=args[0].dtype)
    res = torch.zeros_like(args[0])
    for x in args:
        res += x
    return torch.where(torch.logical_not(torch.isinf(res)), res, zero)




# @torch.jit.script
def Composite_CTC(log_probs: torch.Tensor, # H x W x B x (C+1)
                  log_probs_h: torch.Tensor, # H X B X 2
                  targets: list,  # h x B x n
                  target_lengths: list, # B
                  nb_lines: list, # B
                  input_heights: list = [], # B
                  input_widths: list = [], # B
                  blank: int = 0,
                  lrep: bool = False,
                  reduction: str = 'none', # 'none', 'sum', 'mean'
                  trunc_height: bool = False
                  ):
    _, _, batch_size, nc = log_probs.shape
    B = torch.arange(batch_size, device=log_probs.device)

    log_alpha = forward(log_probs, log_probs_h, targets, target_lengths, nb_lines, input_widths, blank, lrep)

    """实际上忽略了倒数第1行"""
    input_heights = torch.as_tensor(input_heights, dtype=torch.long, device=log_probs.device)
    nb_lines = torch.as_tensor(nb_lines, dtype=torch.long, device=log_probs.device)
    if trunc_height:
        l1l2 = log_alpha[input_heights - 1, B].gather(-1, torch.stack(
            [nb_lines * 2 - 1, nb_lines * 2], dim=-1))
    else:
        l1l2 = log_alpha[- 1, B].gather(-1, torch.stack(
            [nb_lines * 2 - 1, nb_lines * 2], dim=-1))

    loss = -torch.logsumexp(l1l2, dim=-1)

    if reduction.lower() == 'sum':
        loss = torch.sum(loss)
    elif reduction.lower() == 'mean':
        loss = torch.mean(loss)

    return loss



def logadd(*args):
    # produces nan gradients in backward if -inf log-space zero element is used https://github.com/pytorch/pytorch/issues/31829
    res = torch.logsumexp(torch.stack(args), dim=0)
    return res



class LogsumexpFunction(torch.autograd.function.Function):
    @staticmethod
    def forward(self, x0, x1, x2):
        m = torch.max(torch.max(x0, x1), x2)
        m = m.masked_fill_(torch.isinf(m), 0)
        e0 = (x0 - m).exp_()
        e1 = (x1 - m).exp_()
        e2 = (x2 - m).exp_()
        e = (e0 + e1).add_(e2).clamp_(min=1e-16)
        self.save_for_backward(e0, e1, e2, e)
        return e.log_().add_(m)

    @staticmethod
    def backward(self, grad_output):
        e0, e1, e2, e = self.saved_tensors
        g = grad_output / e
        return g * e0, g * e1, g * e2




