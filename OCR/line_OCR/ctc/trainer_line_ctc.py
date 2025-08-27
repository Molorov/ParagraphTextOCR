#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#
#  This software is a computer program written in XXX whose purpose is XXX.
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.

from basic.generic_training_manager import GenericTrainingManager
from basic.utils import edit_wer_from_list, nb_words_from_list, nb_chars_from_list, LM_ind_to_str
from basic.utils import edit_ser_from_list, nb_syllables_from_list, remove_char, nb_comps_from_list
import editdistance
import re
import torch
from torch.nn import CTCLoss
import numpy as np


class TrainerLineCTC(GenericTrainingManager):

    def __init__(self, params):
        super(TrainerLineCTC, self).__init__(params)

    def ctc_remove_successives_identical_ind(self, ind):
        res = []
        for i in ind:
            if res and res[-1] == i:
                continue
            res.append(i)
        return res

    def mctc_remove_successives_identical_ind(self, ind):
        res = []
        for i in range(ind.shape[0]):
            if res and np.all(res[-1] == ind[i]):
                continue
            res.append(ind[i])
        res = np.stack(res, axis=0)
        return res


    def train_batch(self, batch_data, metric_names):
        x = batch_data["imgs"].to(self.device)
        y = batch_data["labels"].to(self.device)
        x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]]
        y_len = batch_data["labels_len"]

        self.optimizer.zero_grad()
        x = self.models["encoder"](x)
        global_pred = self.models["decoder"](x)

        """original ctc loss"""
        loss_ctc = CTCLoss(blank=self.dataset.tokens["blank"], reduction="sum")
        loss = loss_ctc(global_pred.permute(2, 0, 1), y, x_reduced_len, y_len)

        self.backward_loss(loss)
        self.optimizer.step()


        pred = torch.argmax(global_pred, dim=1).cpu().numpy()

        metrics = self.compute_metrics(pred, y.cpu().numpy(), x_reduced_len, y_len, loss=loss.item(), metric_names=metric_names)

        """行预测结果可视化"""
        # vis_folder = os.path.join(self.paths['output_folder'], 'train_line_preds')
        # if not os.path.exists(vis_folder) or len(glob.glob(os.path.join(vis_folder, '*.png'))) < 10:
        #     os.makedirs(vis_folder, exist_ok=True)
        #     ri = random.randrange(0, x.shape[0])
        #     h, w, _ = tuple(batch_data['imgs_shape'][ri])
        #     x_len = x_reduced_len[ri]
        #     img = visualize_line_pred(img=batch_data['imgs'][:, :, 0:h, 0:w][ri],
        #                               log_probs=global_pred[:, :, 0:x_len][ri].detach().cpu(),
        #                               charset=self.dataset.charset, oov_symbols=["•"],
        #                               topk=5,
        #                               topk_pred=5,
        #                               scale=4.,
        #                               opacity=0.5,
        #                               lan=self.params['dataset_params']['lan'])
        #     file_name, ext = os.path.splitext(os.path.basename(batch_data['names'][ri]))
        #     save_name = file_name + f'-epo{self.latest_epoch}' + ext
        #     img = Image.fromarray(img)
        #     img.save(os.path.join(vis_folder, save_name))

        return metrics

    def evaluate_batch(self, batch_data, metric_names, *args):
        x = batch_data["imgs"].to(self.device)
        y = batch_data["labels"].to(self.device)
        x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]]
        y_len = batch_data["labels_len"]

        x = self.models["encoder"](x)
        global_pred = self.models["decoder"](x)


        """original ctc loss"""
        loss_ctc = CTCLoss(blank=self.dataset.tokens["blank"], reduction="sum")
        loss = loss_ctc(global_pred.permute(2, 0, 1), y, x_reduced_len, y_len)


        pred = torch.argmax(global_pred, dim=1).cpu().numpy()

        metrics = self.compute_metrics(pred, y.cpu().numpy(), x_reduced_len, y_len, loss=loss.item(), metric_names=metric_names)

        """只在test阶段输出文本行可视化预测结果"""
        # if self.dataset.test_datasets:
        #     set_name = self.test_set_name
        #     # """取最后一个"""
        #     # for key in self.dataset.test_datasets.keys():
        #     #     set_name = self.dataset.test_datasets[key].set_name
        #     #     # break
        #     load_epoch = self.params['training_params']['load_epoch']
        #     vis_folder = os.path.join(self.paths['output_folder'], f'{set_name}_linepreds_{load_epoch}')

        """最多输出10个"""
        # if not os.path.exists(vis_folder) or len(glob.glob(os.path.join(vis_folder, '*.png'))) < 1000:
        #     os.makedirs(vis_folder, exist_ok=True)
        #     # ri = random.randrange(0, x.shape[0])
        #     for ri in range(x.shape[0]):
        #         # if os.path.basename(batch_data['names'][ri]) == '104.82_5.png':
        #         #     print('dubug')
        #         if metrics['cer'][ri] < 10:
        #             continue
        #         h, w, _ = tuple(batch_data['imgs_shape'][ri])
        #         x_len = x_reduced_len[ri]
        #         if hasattr(self.models['decoder'], 'comp_sizes'):
        #             img = visualize_line_pred_comp(img=batch_data['imgs'][:, :, 0:h, 0:w][ri],
        #                                            pred=pred[ri, :x_len, :],
        #                                            compset=self.dataset.compset,
        #                                            comp_sizes=self.models["decoder"].comp_sizes,
        #                                            opacity=0.5)
        #         else:
        #             img = visualize_line_pred(img=batch_data['imgs'][:, :, 0:h, 0:w][ri],
        #                                       log_probs=global_pred[:, :, 0:x_len][ri].detach().cpu(),
        #                                       charset=self.dataset.charset,
        #                                       charset_mode=self.params['dataset_params']['config_comp']['charset_mode'].lower(),
        #                                       topk=5,
        #                                       opacity=0.5,
        #                                       lan=self.params['dataset_params']['lan'])
        #         file_name, ext = os.path.splitext(os.path.basename(batch_data['names'][ri]))
        #         save_name = file_name + f'-epo{self.latest_epoch}' + ext
        #         img = Image.fromarray(img)
        #         img.save(os.path.join(vis_folder, save_name))

        if "pred" in metric_names:
            # metrics["pred"].extend([batch_data["unchanged_labels"], batch_data["names"]])
            metrics["pred"].extend([batch_data["raw_labels"], batch_data["names"]])
        return metrics

    def compute_metrics(self, x, y, x_len, y_len, loss=None, metric_names=list()):
        batch_size = y.shape[0]
        ind_x = [x[i][:x_len[i]] for i in range(batch_size)]
        ind_y = [y[i][:y_len[i]] for i in range(batch_size)]
        """debug: 保存原始预测序列"""
        # blank字符是“•”

        """字丁级识别"""

        ind_x = [self.ctc_remove_successives_identical_ind(t) for t in ind_x]
        if self.params['dataset_params']['text_type'] == 'list':
            # 藏文数据集
            str_x = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="", return_list=True) for t in ind_x]
            str_y = [LM_ind_to_str(self.dataset.charset, t, return_list=True) for t in ind_y]
            """对于藏文，空格的预测可有可无，因此全部删掉"""
            str_x = [remove_char(t, remove=[' ']) for t in str_x]
            str_y = [remove_char(t, remove=[' ']) for t in str_y]

        else:
            str_x = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="") for t in ind_x]
            str_y = [LM_ind_to_str(self.dataset.charset, t) for t in ind_y]
            str_x = [re.sub("( )+", ' ', t).strip(" ") for t in str_x]

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
            elif metric_name == "pred":
                metrics["pred"] = [str_x, ]
            # elif metric_name == "raw_pred":
            #     metrics["raw_pred"] = [raw_pred, ]
        if "loss_ctc" in metric_names:
            metrics["loss_ctc"] = loss / metrics["nb_chars"]
        metrics["nb_samples"] = len(x)
        return metrics
