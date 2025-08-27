from basic.generic_training_manager import GenericTrainingManager
import torch
from basic.utils import edit_wer_from_list, nb_chars_from_list, nb_words_from_list, LM_ind_to_str, nb_comps_from_list
import numpy as np
import editdistance
import re
from basic.utils import remove_successive_spaces_bo, strip_bo, edit_ser_from_list, nb_syllables_from_list
from Loss.composite_ctc import Composite_CTC
from OCR.document_OCR.comp_ctc.models_pg_cc import DecoderH
from basic.line_match import match_lines
from visulization.visualize import visualize_pred
import os

"""Training manager for composite-CTC loss"""
class Manager_CC(GenericTrainingManager):

    def __init__(self, params):
        super(Manager_CC, self).__init__(params)

    def train_batch(self, batch_data, metric_names, *args):
        self.optimizer.zero_grad()

        x = batch_data["imgs"].to(self.device)
        y = [l.to(self.device) for l in batch_data["line_labels"]]
        y_len = batch_data["line_labels_len"]
        x_reduced_width = [s[1] for s in batch_data["imgs_reduced_shape"]]
        x_reduced_height = [s[0] for s in batch_data["imgs_reduced_shape"]]
        nb_lines = batch_data['nb_lines']

        batch_size = y[0].size()[0]

        features = self.models["encoder"](x)
        if 'attention' in self.models.keys():
            features, attn_weights = self.models['attention'](features)
        if isinstance(self.models['decoder'], DecoderH):
            log_probs, log_probs_h = self.models['decoder'](features)
        else:
            log_probs = self.models['decoder'](features)
            log_probs_h = None


        line_repeat = self.params['model_params']['comp_ctc'].get('line_repeat', False)
        trunc_height = self.params['model_params']['comp_ctc'].get('trunc_height', False)
        blank = self.dataset.tokens["blank"]
        log_probs_h_ = log_probs_h.permute(2, 0, 1) if log_probs_h is not None else None

        loss_comp_ctc = Composite_CTC(log_probs.permute(2, 3, 0, 1), log_probs_h_, y, y_len, nb_lines, x_reduced_height, x_reduced_width,
                                      blank=blank, lrep=line_repeat, reduction='sum', trunc_height=trunc_height)


        self.backward_loss(loss_comp_ctc)
        self.optimizer.step()

        global_pred = torch.argmax(log_probs, dim=1).cpu().numpy()
        # global_prob = torch.max(log_probs, dim=1).values.detach().exp().cpu().numpy()

        lines_pred = []
        if log_probs_h is not None:
            global_pred_h = torch.argmax(log_probs_h, dim=1)
            for i in range(batch_size):
                preds = global_pred[i, :, :x_reduced_width[i]]
                line_pred = []
                height = x_reduced_height[i] if trunc_height else preds.shape[0]
                for j in range(height):
                    if global_pred_h[i, j] == 0:
                        line_pred.append(preds[j, :])
                lines_pred.append(line_pred)
        else:
            for i in range(batch_size):
                if trunc_height:
                    line_pred = [global_pred[i, j, :x_reduced_width[i]] for j in range(x_reduced_height[i])]
                else:
                    line_pred = [global_pred[i, j, :x_reduced_width[i]] for j in range(global_pred.shape[1])]
                lines_pred.append(line_pred)

        metrics = self.compute_metrics(lines_pred, batch_data["raw_labels"], metric_names, from_line=True)
        if "loss_ctc" in metric_names:
            metrics["loss_ctc"] = loss_comp_ctc.item() / metrics["nb_chars"]
        if "diff_len" in metric_names:
            diff_len = np.array(metrics['nb_lines']) - np.array(batch_data["nb_lines"])
            metrics["diff_len"] = diff_len
        del metrics['nb_lines']
        return metrics

    def evaluate_batch(self, batch_data, metric_names, ema=False, line_match=False, visualize=False):
        x = batch_data["imgs"].to(self.device)
        y = [l.to(self.device) for l in batch_data["line_labels"]]
        y_len = batch_data["line_labels_len"]
        x_reduced_width = [s[1] for s in batch_data["imgs_reduced_shape"]]
        x_reduced_height = [s[0] for s in batch_data["imgs_reduced_shape"]]
        nb_lines = batch_data['nb_lines']

        batch_size = y[0].size()[0]

        models = self.models_ema if ema else self.models

        features = models["encoder"](x)
        if 'attention' in models.keys():
            features, attn_weights = models['attention'](features)
        if isinstance(models['decoder'], DecoderH):
            log_probs, log_probs_h = models['decoder'](features)
        else:
            log_probs = models['decoder'](features)
            log_probs_h = None

        line_repeat = self.params['model_params']['comp_ctc'].get('line_repeat', False)
        trunc_height = self.params['model_params']['comp_ctc'].get('trunc_height', False)
        blank = self.dataset.tokens["blank"]
        log_probs_h_ = log_probs_h.permute(2, 0, 1) if log_probs_h is not None else None

        if "loss_ctc" in metric_names:
            loss_comp_ctc = Composite_CTC(log_probs.permute(2, 3, 0, 1), log_probs_h_, y, y_len, nb_lines, x_reduced_height,
                                          x_reduced_width,
                                          blank=blank, lrep=line_repeat, reduction='sum', trunc_height=trunc_height)



        global_pred = torch.argmax(log_probs, dim=1).cpu().numpy()
        # global_prob = torch.max(log_probs, dim=1).values.detach().exp().cpu().numpy()

        lines_pred = []
        if log_probs_h is not None and not line_match:
            global_pred_h = torch.argmax(log_probs_h, dim=1)
            global_prob_h = torch.max(log_probs_h, dim=1).values.detach().exp().cpu().numpy()
            conf_thres_h = self.params['model_params']['models']['decoder'].get('conf_thres', 0.5)
            for i in range(batch_size):
                preds = global_pred[i, :, :x_reduced_width[i]]
                line_pred = []
                height = x_reduced_height[i] if trunc_height else preds.shape[0]
                for j in range(height):
                    if global_pred_h[i, j] == 0 and global_prob_h[i, j] >= conf_thres_h:
                        line_pred.append(preds[j, :])
                lines_pred.append(line_pred)
        else:
            for i in range(batch_size):
                if trunc_height:
                    line_pred = [global_pred[i, j, :x_reduced_width[i]] for j in range(x_reduced_height[i])]
                else:
                    line_pred = [global_pred[i, j, :x_reduced_width[i]] for j in range(global_pred.shape[1])]
                lines_pred.append(line_pred)

        metrics = self.compute_metrics(lines_pred, batch_data["raw_labels"], metric_names, from_line=True, line_match=line_match, lines_raw=batch_data['line_raw'])
        if "loss_ctc" in metric_names:
            metrics["loss_ctc"] = loss_comp_ctc.item() / metrics["nb_chars"]
        if "diff_len" in metric_names:
            diff_len = np.array(metrics['nb_lines']) - np.array(batch_data["nb_lines"])
            # if np.any(diff_len):
            #     print('difference in lengths')
            metrics["diff_len"] = diff_len
        del metrics['nb_lines']

        if visualize:
            # assert len(batch_data['names']) == 1
            set_name = self.test_set_name
            output_folder = os.path.join(self.paths['output_folder'], 'visualization', f'{set_name}')
            os.makedirs(output_folder, exist_ok=True)
            for i in range(x.shape[0]):
                H, W, _ = tuple(batch_data['imgs_shape'][i])
                h, w, _ = tuple(batch_data['imgs_reduced_shape'][i])
                img = visualize_pred(batch_data['imgs'][i, :, :, :],
                                     global_pred[i, :, :],
                                     global_pred_h[i, :] if log_probs_h is not None else None,
                                     None, # attn_weights[i].detach().cpu().numpy(),
                                     self.dataset.charset)
                img_name = os.path.basename(batch_data['names'][i])
                # cv2.imwrite(os.path.join(output_folder, img_name), img)
                img.save(os.path.join(output_folder, img_name))
        return metrics

    def ctc_remove_successives_identical_ind(self, ind):
        res = []
        for i in ind:
            if res and res[-1] == i:
                continue
            res.append(i)
        return res

    def compute_metrics(self, ind_x, str_y, metric_names=list(), from_line=False, line_match=False, lines_raw=None):
        if from_line:
            str_x = list()
            nb_lines = []
            for i, lines_token in enumerate(ind_x):
                if self.params['dataset_params']['text_type'] == 'list':
                    """藏文数据集"""
                    list_str = [LM_ind_to_str(self.dataset.charset, self.ctc_remove_successives_identical_ind(p),
                                              oov_symbol="", return_list=True) if p is not None else [] for p in lines_token]
                    list_str = [remove_successive_spaces_bo(p) for p in list_str]
                    list_str = [strip_bo(p, remove_chars=[' ', '\xa0']) for p in list_str]
                    if line_match:
                        list_str, _, _, _, _, _ = match_lines(lines_raw[i], list_str)
                    list_str_full = []
                    nb_line = 0
                    for line in list_str:
                        # 忽略空行
                        if len(line) > 0:
                            nb_line += 1
                            if len(list_str_full):  # 非首行
                                # 行和行之间以' '做间隔
                                list_str_full += [' ']
                            list_str_full += line
                    str_x.append(list_str_full)
                    nb_lines.append(nb_line)
                else:
                    list_str = [LM_ind_to_str(self.dataset.charset, self.ctc_remove_successives_identical_ind(p),
                                              oov_symbol="") if p is not None else "" for p in lines_token]
                    if line_match:
                        list_str, _, _, _, _, _ = match_lines(lines_raw[i], list_str)
                    str_x.append(re.sub("( )+", ' ', " ".join(list_str).strip(" ")))
                    nb_lines.append(np.sum([1 if len(l) > 0 else 0 for l in list_str]))
        else:
            if self.params['dataset_params']['text_type'] == 'list':
                """
                藏文数据集
                """
                list_str = [LM_ind_to_str(self.dataset.charset, self.ctc_remove_successives_identical_ind(p),
                                          oov_symbol="", lan='bo') if p is not None else [] for p in ind_x]
                list_str = [remove_successive_spaces_bo(p) for p in list_str]
                list_str = [strip_bo(p, remove_chars=[' ', '\xa0']) for p in list_str]
                str_x = list_str
            else:
                str_x = [LM_ind_to_str(self.dataset.charset, self.ctc_remove_successives_identical_ind(p),
                                       oov_symbol="") if p is not None else "" for p in ind_x]
        metrics = dict()
        for metric_name in metric_names:
            if metric_name == 'coer':
                metrics[metric_name] = [editdistance.eval(''.join(u), ''.join(v)) for u, v in zip(str_y, str_x)]
                metrics['nb_comps'] = nb_comps_from_list(str_y)
            elif metric_name == "cer":
                metrics[metric_name] = [editdistance.eval(u, v) for u, v in zip(str_y, str_x)]
                metrics["nb_chars"] = nb_chars_from_list(str_y)
            elif metric_name == "wer":
                if self.params['dataset_params']['text_type'] == 'list':
                    # 藏文计算SER（Syllable Error Rate）
                    metrics[metric_name] = edit_ser_from_list(str_y, str_x)
                    metrics["nb_words"] = nb_syllables_from_list(str_y)
                else:
                    metrics[metric_name] = edit_wer_from_list(str_y, str_x)
                    metrics["nb_words"] = nb_words_from_list(str_y)
            # elif metric_name == "pred":
            #     metrics["pred"] = [preds]
            # elif metric_name == "raw_pred":
            #     metrics["raw_pred"] = [raw_preds]

        metrics["nb_samples"] = len(str_x)
        metrics['nb_lines'] = nb_lines
        return metrics
