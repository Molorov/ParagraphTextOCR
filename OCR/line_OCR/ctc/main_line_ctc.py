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

import copy
import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.insert(0, '../../..')
from OCR.line_OCR.ctc.trainer_line_ctc import TrainerLineCTC
from OCR.line_OCR.ctc.models_line_ctc import Decoder
from basic.models import FCN_Encoder
from torch.optim import Adam
from basic.generic_dataset_manager import OCRDataset
import torch.multiprocessing as mp
import torch
import yaml
import argparse


def train_and_test(rank, params, test):
    params["training_params"]["ddp_rank"] = rank
    model = TrainerLineCTC(params)
    # Model trains until max_time_training or max_nb_epochs is reached
    if not test:
        model.train()

    # load weights giving best CER on valid set
    model.params["training_params"]["load_epoch"] = "best"
    model.load_model()

    """
    作为跨域实验的test数据
    """
    if 'extra_test' in params['dataset_params']:
        for dataset_name in params['dataset_params']['extra_test']:
            params['dataset_params']['datasets'][dataset_name] = os.path.join(data_root, 'formatted',
                                                                                   dataset_name + '_lines')


    # compute metrics on train, valid and test sets (in eval conditions)
    metrics = ["cer", "wer", "time", "worst_cer", "pred", 'coer']
    for dataset_name in params["dataset_params"]["datasets"].keys():
        """用于test的跨域数据集"""
        if 'extra_test' in params['dataset_params'] and dataset_name in params['dataset_params']['extra_test']:
            """保存原来的config"""
            old_config = copy.deepcopy(params['dataset_params']['config'])
            """将原来的dataset的config替换成跨域数据集的config"""
            # params['dataset_params']['config']
            for config_term in params['dataset_params']['config'].keys():
                if config_term in params['dataset_params']['extra_test'][dataset_name].keys():
                    params['dataset_params']['config'][config_term] = params['dataset_params']['extra_test'][dataset_name][config_term]

        for set_name in ["test", "valid", "train"]:
            model.predict("{}-{}".format(dataset_name, set_name), [(dataset_name, set_name), ], metrics)

        if 'extra_test' in params['dataset_params'] and dataset_name in params['dataset_params']['extra_test']:
            """恢复原来的dataset的config"""
            params['dataset_params']['config'] = old_config



parser = argparse.ArgumentParser(description='Generic runner for CTC')

parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help = 'path to the config file')

parser.add_argument('--data', '-d',
                    dest="data_root",
                    default='../../../Datasets')

parser.add_argument('--test', '-t',
                    dest="test",
                    action='store_true',
                    default=False)

args = parser.parse_args()
with open(args.filename, 'r', encoding='utf-8') as file:
    try:
        params = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# data_root = '../../../Datasets'
# data_root = 'c:/myDatasets/VAN'
# 服务器上用这个
# data_root = '../../../../VAN-data'

data_root = args.data_root

if __name__ == '__main__':
    if 'custom_ctc' in params['model_params']: # and 'edge-ctc' in params['model_params']['custom_ctc']['method']:
        params['dataset_params']['config']['charset_mode'] = params['model_params']['custom_ctc']['method']
    else:
        params['dataset_params']['config']['charset_mode'] = 'CTC'

    dataset_name = params['dataset_params']['name']

    params['dataset_params']['datasets'] = {dataset_name: os.path.join(data_root, 'formatted', dataset_name + '_lines')}
    params['dataset_params']['train'] = {
        'name': "{}-train".format(dataset_name),
        'datasets': [dataset_name]
    }
    params['dataset_params']['valid'] = {
        "{}-valid".format(dataset_name): [dataset_name],
    }


    params['dataset_params']['dataset_class'] = globals()[params['dataset_params']['dataset_class']]

    for model_key, model_param in params['model_params']['models'].items():
        if type(model_param) is str:
            params['model_params']['models'][model_key] = globals()[model_param] # mondule_param is the name
        else: # dict
            params['model_params']['models'][model_key]['name'] = globals()[model_param['name']]

    if 'dropout_scheduler' in params['model_params']:
        params['model_params']['dropout_scheduler']['function'] = globals()[params['model_params']['dropout_scheduler']['function']]


    params['training_params']['output_folder'] = os.path.splitext(os.path.basename(args.filename))[0]
    params['training_params']['max_training_time'] = 3600 * (24+23)
    params['training_params']['nb_gpu'] = torch.cuda.device_count()
    params['training_params']['optimizer']['class'] = globals()[params['training_params']['optimizer']['class']]
    params['training_params']['set_name_focus_metric'] = "{}-valid".format(dataset_name)

    if params["training_params"]["use_ddp"] and not params["training_params"]["force_cpu"]:
        mp.spawn(train_and_test, args=(params,), nprocs=params["training_params"]["nb_gpu"])
    else:
        train_and_test(0, params, args.test)

