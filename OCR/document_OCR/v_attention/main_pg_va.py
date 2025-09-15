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
import os
import sys
sys.path.insert(0, '../../..')
from torch.optim import Adam
from OCR.document_OCR.v_attention.models_pg_va import VerticalAttention, LineDecoderCTC
from OCR.document_OCR.v_attention.trainer_pg_va import Manager
from basic.models import FCN_Encoder
from basic.generic_dataset_manager import OCRDataset
import torch
import torch.multiprocessing as mp


def train_and_test(rank, params, test, load_epoch, line_match):
    params["training_params"]["ddp_rank"] = rank
    model = Manager(params)
    # Model trains until max_time_training or max_nb_epochs is reached
    if not test:
        model.train()


    model.params["training_params"]["load_epoch"] = 'best' if 'best' in load_epoch else 'last'
    model.load_model()
    ema = True if 'ema' in load_epoch else False

    # compute metrics on train, valid and test sets (in eval conditions)
    metrics = ["cer", "wer", "diff_len", "time", "worst_cer"]
    for dataset_name in params["dataset_params"]["datasets"].keys():
        for set_name in ['test', "valid", "train"]:
            model.predict("{}-{}".format(dataset_name, set_name), [(dataset_name, set_name), ], metrics, ema, line_match)

import argparse
import yaml

parser = argparse.ArgumentParser(description='Generic runner')

parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help = 'path to the config_comp file')

parser.add_argument('--data_root', '-d',
                    dest="data_root",
                    default='../../../Datasets')

parser.add_argument('--test', '-t',
                    dest="test",
                    action='store_true',
                    default=False)

parser.add_argument('--epoch', '-e',
                    dest="load_epoch",
                    # action='store_true',
                    default='best')

parser.add_argument('--line_match', '-lm',
                    dest="line_match",
                    action='store_true',
                    default=False)

args = parser.parse_args()
with open(args.filename, 'r', encoding='utf-8') as file:
    try:
        params = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

data_root = args.data_root


if __name__ == '__main__':
    dataset_name = params['dataset_params']['name']

    params['dataset_params']['datasets'] = {dataset_name: os.path.join(data_root, 'formatted', dataset_name + '_paragraph')}
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

    # params['training_params']['manager'] = globals()[params['training_params'].get('manager', 'Manager')]
    params['training_params']['output_folder'] = os.path.splitext(os.path.basename(args.filename))[0]
    params['training_params']['max_training_time'] = 3600 * (24 + 23)
    params['training_params']['nb_gpu'] = torch.cuda.device_count()
    params['training_params']['optimizer']['class'] = globals()[params['training_params']['optimizer']['class']]
    params['training_params']['set_name_focus_metric'] = "{}-valid".format(dataset_name)

    if params["training_params"]["stop_mode"] == "learned":
        params["training_params"]["train_metrics"].append("loss_ce")
    params["model_params"]["stop_mode"] = params["training_params"]["stop_mode"]

    if params["training_params"]["use_ddp"] and not params["training_params"]["force_cpu"]:
        mp.spawn(train_and_test, args=(params,), nprocs=params["training_params"]["nb_gpu"])
    else:
        train_and_test(0, params, args.test, args.load_epoch, args.line_match)



