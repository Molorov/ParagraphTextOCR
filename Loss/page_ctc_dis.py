import torch
import torch.nn.functional as F
from torch import Tensor as Tensor
from torch import tensor as tensor
from torch.nn import CTCLoss

finfo_min_fp32 = torch.finfo(torch.float32).min
finfo_min_fp16 = torch.finfo(torch.float16).min


# @ torch.jit.script
# def logadd(*args):
#     return torch.logsumexp(torch.stack(args), dim=0)

@torch.jit.script
def logadd(arg1, arg2, arg3):
    return torch.logsumexp(torch.stack([arg1, arg2, arg3]), dim=0)


@torch.jit.script
def forward_dis(log_probs: Tensor,  # (T, B, 2) flattend and maybe truncated!!
                targets: Tensor,  # (B, max_len)
                blank: int,
                zero: Tensor
                ):
    input_time_len, batch_size, num_class = log_probs.shape

    _t_a_r_g_e_t_s_ = torch.cat([targets, targets[:, :1]], dim=1)
    _t_a_r_g_e_t_s_ = torch.stack([
        torch.full_like(_t_a_r_g_e_t_s_, blank),
        _t_a_r_g_e_t_s_
    ], dim=-1).flatten(start_dim=-2)
    _t_a_r_g_e_t_s_ = _t_a_r_g_e_t_s_[:, :-1]  # (B, 2*max_len+1)

    log_probs_ = log_probs.gather(-1, _t_a_r_g_e_t_s_.expand(input_time_len, -1, -1))

    padding = 2
    log_alpha = torch.full((input_time_len, batch_size, padding + _t_a_r_g_e_t_s_.shape[-1]),
                           fill_value=zero, dtype=log_probs.dtype, device=log_probs.device)
    log_alpha[0, :, padding + 0] = log_probs_[0, :, 0]
    log_alpha[0, :, padding + 1] = log_probs_[0, :, 1]

    is_char = _t_a_r_g_e_t_s_ < blank
    is_blank = _t_a_r_g_e_t_s_ == blank
    for t in range(1, input_time_len):
        log_alpha[t, :, 2:] = log_probs_[t, :, :] + logadd(torch.where(is_blank, log_alpha[t - 1, :, 2:], zero),
                                                           log_alpha[t - 1, :, 1:-1],
                                                           torch.where(is_char, log_alpha[t - 1, :, :-2], zero))
    return log_alpha[:, :, padding:]


# @ torch.jit.script
def forward(log_probs_cls: Tensor,  # (H, W, nbl, |A|)
            log_probs_dis: Tensor, # (H, W, nbl, 2)
            log_probs_ro: Tensor,  # (H, W, nbl, 9)
            targets: Tensor,  # (nbl, max_len)
            zero: Tensor
            ):
    input_time_len, input_spatial_len, nb_lines, num_class = log_probs_cls.shape
    _, max_len = targets.shape

    padding = 1
    blank = num_class + 0
    log_probs_cls_ = log_probs_cls + log_probs_dis[:, :, :, :1].expand(-1, -1, -1, num_class)
    log_probs_cls_ = torch.cat([log_probs_cls_,
                                log_probs_dis[:, :, :, 1:]], dim=-1)

    expanded_targets = torch.stack([targets, torch.full_like(targets, blank)], dim=-1).flatten(
        start_dim=-2)
    expanded_targets = expanded_targets[:, :-1]  # (nb_lines, 2*maxlen-1)


    log_probs = log_probs_cls_.gather(-1, expanded_targets.expand(input_time_len, input_spatial_len, -1, -1))

    # log_probs_ro_ = log_probs_ro.unsqueeze(dim=3).expand(-1, -1, -1, expanded_targets.shape[-1],  -1)

    log_alpha = torch.full((input_time_len, input_spatial_len, nb_lines, padding + expanded_targets.shape[-1]),
                           fill_value=zero, dtype=log_probs_cls.dtype, device=log_probs_cls.device)

    log_alpha[:, :, :, 1] = log_probs[:, :, :, 0] + log_probs_ro[:, :, :, 4]

    log_probs_ = log_probs[:, :, :, 1:]
    expanded_targets_ = expanded_targets.expand(input_spatial_len, -1, -1)
    expanded_targets_ = expanded_targets_[:, :, 1:]

    is_char = expanded_targets_ < num_class
    is_blank = expanded_targets_ >= num_class

    # alpha = log_alpha[:, :, 0, :].permute(1, 2, 0).detach().cpu().numpy()
    zero_pad_spatial = torch.full((1, nb_lines, log_probs_.shape[-1]), fill_value=zero,
                                  dtype=log_probs.dtype, device=log_probs.device)
    for t in range(1, input_time_len):
        to_add = logadd(torch.where(is_blank, log_alpha[t - 1, :, :, 2:], zero),
                        log_alpha[t - 1, :, :, 1:-1],
                        torch.where(is_char, log_alpha[t - 1, :, :, :-2], zero))
        to_add = torch.cat([zero_pad_spatial, to_add, zero_pad_spatial], dim=0)
        log_probs_ro_t = log_probs_ro[t].unsqueeze(dim=2).expand(-1, -1, expanded_targets_.shape[-1], -1)
        log_alpha[t, :, :, 2:] = log_probs_[t] + logadd(to_add[:-2] + log_probs_ro_t[:, :, :, 0],
                                                        to_add[1:-1] + log_probs_ro_t[:, :, :, 3],
                                                        to_add[2:] + log_probs_ro_t[:, :, :, 6])

    return log_alpha[:, :, :, padding:]


def page_ctc_dis(log_probs_cls: Tensor,  # (H, W, B, |A|)
                 log_probs_dis: Tensor,  # (H, W, B, 2)
                 log_probs_ro: Tensor,  # (H, W, B, 9)
                 targets: Tensor,  # (max_nb_line, B, max_len)
                 target_lengths: list,  # B
                 nb_lines: list,  # B
                 input_heights: list = [],  # B
                 input_widths: list = [],  # B
                 reduction: str = 'none',  # 'none', 'sum', 'mean'
                 trunc_width: bool = False,
                 trunc_height: bool = False,
                 directions: list = ['horizontal']
                 ):
    H, W, batch_size, nc = log_probs_cls.shape
    B = torch.arange(batch_size, device=log_probs_cls.device)
    zero = tensor(finfo_min_fp16 if log_probs_cls.dtype == torch.float16 else finfo_min_fp32,
                  device=log_probs_cls.device, dtype=log_probs_cls.dtype)
    """处理pad字符"""
    targets = torch.where(targets < log_probs_cls.shape[-1], targets, 0)

    # nb_chars = torch.sum(torch.as_tensor(target_lengths, dtype=targets.dtype, device=targets.device), dim=0)
    # targets_dis = torch.full((batch_size, torch.max(nb_chars)), fill_value=0, dtype=targets.dtype, device=targets.device)
    #
    # input_lens = []
    # log_flat_dis = torch.full((log_probs_dis.shape[0]*log_probs_dis.shape[1], batch_size, 2), fill_value=0.,
    #                           dtype=log_probs_cls.dtype,
    #                           device=log_probs_cls.device)
    # for b in range(batch_size):
    #     height = input_heights[b] if trunc_height else log_probs_dis.shape[0]
    #     width = input_widths[b] if trunc_width else log_probs_dis.shape[1]
    #     log_flat_dis[:height*width, b, :] = log_probs_dis[:height, :width, b, :].flatten(start_dim=0, end_dim=1)
    #     input_lens.append(height*width)
    # input_lens = torch.as_tensor(input_lens, dtype=torch.long, device=log_probs_cls.device)
    # log_flat_dis = log_flat_dis[:torch.max(input_lens), :, :]
    """使用NativeCTC限制Odis的数量？"""
    # l1l2_dis = - F.ctc_loss(log_flat_dis, targets_dis, input_lens, nb_chars, blank=1, reduction='none')
    """使用CompactCTC限制Odis的数量"""
    # log_alpha_dis = forward_dis(log_flat_dis, targets_dis, blank=1, zero=zero)
    # l1l2_dis = log_alpha_dis[input_lens - 1, B].gather(-1, torch.stack(
    #     [nb_chars * 2 - 1, nb_chars * 2], dim=-1))
    # l1l2_dis = torch.logsumexp(l1l2_dis, dim=-1)

    # if log_probs_dis is None:
    #     log_probs_dis = torch.zeros((H, W, batch_size, 2), dtype=log_probs_cls.dtype, device=log_probs_cls.device)

    log_probs_cls_ = []
    log_probs_dis_ = []
    log_probs_ro_ = []
    targets_ = []
    targets_length_ = []
    input_heights_ = []
    input_widths_ = []
    total_lens = []
    for b in range(batch_size):
        log_probs_cls_.append(log_probs_cls[:, :, b:b + 1, :].expand(-1, -1, nb_lines[b], -1))
        log_probs_dis_.append(log_probs_dis[:, :, b:b + 1, :].expand(-1, -1, nb_lines[b], -1))
        log_probs_ro_.append(log_probs_ro[:, :, b:b + 1, :].expand(-1, -1, nb_lines[b], -1))
        targets_.append(targets[:nb_lines[b], b, :])
        targets_length_ += [target_lengths[i][b] for i in range(nb_lines[b])]
        total_lens.append(sum([target_lengths[i][b] for i in range(nb_lines[b])]))
        input_heights_ += [input_heights[b]] * nb_lines[b]
        input_widths_ += [input_widths[b]] * nb_lines[b]
    log_probs_cls_ = torch.cat(log_probs_cls_, dim=2)
    log_probs_dis_ = torch.cat(log_probs_dis_, dim=2) if log_probs_dis is not None else None
    log_probs_ro_ = torch.cat(log_probs_ro_, dim=2)
    targets_ = torch.cat(targets_, dim=0)
    targets_length_ = torch.as_tensor(targets_length_, dtype=targets.dtype, device=targets.device)
    input_heights_ = torch.as_tensor(input_heights_, dtype=targets.dtype, device=targets.device)
    input_widths_ = torch.as_tensor(input_widths_, dtype=targets.dtype, device=targets.device)

    nb_lines_total = log_probs_cls_.shape[2]

    l1l2 = []
    for direction in directions:
        assert direction in ['horizontal', 'vertical']
        if direction == 'horizontal':
            log_alpha = forward(log_probs_cls_.permute(1, 0, 2, 3), log_probs_dis_.permute(1, 0, 2, 3),
                                log_probs_ro_.permute(1, 0, 2, 3), targets_, zero)
            time_len = input_widths_ if trunc_width else [log_alpha.shape[0]] * nb_lines_total
            spatial_len = input_heights_ if trunc_height else [log_alpha.shape[1]] * nb_lines_total
        elif direction == 'vertical':
            log_alpha = forward(log_probs_cls_,  log_probs_dis_,
                                log_probs_ro_, targets_, zero)
            time_len = input_heights_ if trunc_height else [log_alpha.shape[0]] * nb_lines_total
            spatial_len = input_widths_ if trunc_width else [log_alpha.shape[1]] * nb_lines_total

        # alpha = log_alpha[:, :, 0, :].permute(1, 2, 0).detach().cpu().numpy()

        log_alpha_ = [
            log_alpha[:time_len[l], :spatial_len[l], l, targets_length_[l] * 2 - 2]
            for l in range(log_probs_cls_.shape[2])]
        l1l2.append(torch.stack([torch.logsumexp(la, dim=(0, 1)) for la in log_alpha_]))

    l1l2 = torch.logsumexp(torch.stack(l1l2, dim=0), dim=0)

    loss = []
    n = 0
    for b in range(batch_size):
        loss_batch = - (torch.sum(l1l2[n: n + nb_lines[b]]))  # + l1l2_dis[b])
        loss.append(loss_batch)
        n += nb_lines[b]

    loss = torch.stack(loss)

    if reduction.lower() == 'sum':
        loss = torch.sum(loss)
    elif reduction.lower() == 'mean':
        loss = torch.mean(loss)

    return loss