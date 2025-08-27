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

import editdistance
import re
import numpy as np

# Charset / labels conversion

def LM_str_to_ind(labels, str):
    return [labels.index(c) for c in str]



def LM_ind_to_str(labels, ind, oov_symbol=None, return_list=False):
    if oov_symbol is not None:
        res = []
        for i in ind:
            if i < len(labels):
                # 正常字符，包括空格
                res.append(labels[i])
            else:
                if len(oov_symbol):
                    symbol = oov_symbol
                    res.append(symbol)
                # blank 字符
                # 藏文label由于是list, 所以在oov_symbol是空字符串的情况下，不需要添加
                # if lan != 'bo' or oov_symbol != '':
                #     res.append(oov_symbol)
    else:
        res = [labels[i] for i in ind]
    if return_list:
        # 藏文
        return res
    else:
        return "".join(res)


# OCR METRICS


def edit_cer_from_list(truth, pred):
    edit = 0
    for t, p in zip(truth, pred):
        edit += editdistance.eval(t, p)
    return edit


def format_string_for_wer(str):
    str = re.sub('([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ', str)
    str = re.sub('([ \n])+', " ", str).strip()
    return str

def format_char_list_for_ser(char_list):
    res = list()
    """ 
    chr(0xF0B) -- tsek
    chr(0xF0D) -- 单垂线
    chr(0xF0E) -- 双垂线
    """
    deliminators = ['\n', ' ', '\xa0', chr(0xF0B), chr(0xF0D), chr(0xF0E)]
    syllable = ''
    for c in char_list:
        if syllable and c in deliminators:
            res.append(syllable)
            syllable = ''
            continue
        if c not in deliminators:
            syllable += c
    if syllable:
        res.append(syllable)
    return res

def edit_wer_from_list(truth, pred):
    edit = 0
    for pred, gt in zip(pred, truth):
        gt = format_string_for_wer(gt)
        pred = format_string_for_wer(pred)
        gt = gt.split(" ")
        pred = pred.split(" ")
        edit += editdistance.eval(gt, pred)
    return edit


def nb_words_from_list(list_gt):
    len_ = 0
    for gt in list_gt:
        gt = format_string_for_wer(gt)
        gt = gt.split(" ")
        len_ += len(gt)
    return len_

def edit_ser_from_list(truth, pred):
    edit = 0
    for pred_, gt in zip(pred, truth):
        gt = format_char_list_for_ser(gt)
        pred_ = format_char_list_for_ser(pred_)
        edit += editdistance.eval(gt, pred_)
    return edit

def nb_syllables_from_list(list_gt):
    len_ = 0
    for gt in list_gt:
        gt = format_char_list_for_ser(gt)
        len_ += len(gt)
    return len_


def nb_chars_from_list(list_gt):
    return sum([len(t) for t in list_gt])

def nb_comps_from_list(list_gt):
    return sum([len(''.join(t)) for t in list_gt])


def cer_from_list_str(str_gt, str_pred):
        len_ = 0
        edit = 0
        for pred, gt in zip(str_pred, str_gt):
            edit += editdistance.eval(gt, pred)
            len_ += len(gt)
        cer = edit / len_
        return cer


def wer_from_list_str(str_gt, str_pred):
    len_ = 0
    edit = 0
    for pred, gt in zip(str_pred, str_gt):
        gt = format_string_for_wer(gt)
        pred = format_string_for_wer(pred)
        gt = gt.split(" ")
        pred = pred.split(" ")
        edit += editdistance.eval(gt, pred)
        len_ += len(gt)
    cer = edit / len_
    return cer


def cer_from_files(file_gt, file_pred):
        with open(file_pred, "r") as f_p:
            str_pred = f_p.readlines()
        with open(file_gt, "r") as f_gt:
            str_gt = f_gt.readlines()
        return cer_from_list_str(str_gt, str_pred)


def wer_from_files(file_gt, file_pred):
    with open(file_pred, "r") as f_p:
        str_pred = f_p.readlines()
    with open(file_gt, "r") as f_gt:
        str_gt = f_gt.readlines()
    return wer_from_list_str(str_gt, str_pred)


def remove_char(char_list, remove):
    return [c for c in char_list if c not in remove]


def remove_successive_spaces_bo(text):
    if type(text) is dict:
        """部件级文本"""
        new_text = {comp_id: [] for comp_id in text.keys()}
        for i in range(len(text['HPC'])):
            if new_text['HPC'] and new_text['HPC'][-1] in [' ', '\xa0'] and text['HPC'][i] in [' ', '\xa0']:
                continue
            for comp_id in new_text.keys():
                new_text[comp_id].append(text[comp_id][i])
    elif type(text) is list:
        new_text = list()
        for c in text:
            # '\xa0' -- none-breaking space
            if new_text and new_text[-1] in [' ', '\xa0'] and c in [' ', '\xa0']:
                continue
            new_text.append(c)
    else:
        raise TypeError
    return new_text

def replace_bo(text, map_dict):
    if type(text) is dict:
        """部件级文本, 只替换基字"""
        new_text = {comp_id: [] for comp_id in text.keys()}
        for i in range(len(text['HPC'])):
            if text['HPC'][i] in map_dict.keys():
                if map_dict[text['HPC'][i]] == '':
                    """替换部件为''时，直接删除该字丁"""
                    continue
                new_text['HPC'].append(map_dict[text['HPC'][i]])
                for comp_id in new_text.keys():
                    if comp_id != 'HPC':
                        assert text[comp_id][i] == ''
                        new_text[comp_id].append(text[comp_id][i])
            else:
                for comp_id in new_text.keys():
                    new_text[comp_id].append(text[comp_id][i])
    elif type(text) is list:
        return NotImplementedError
    else:
        return TypeError
    return new_text



def strip_bo(text, remove_chars):
    start = 0
    if type(text) is dict:
        """部件级文本"""
        end = len(text['HPC'])
        for i in range(len(text['HPC'])):
            if text['HPC'][i] not in remove_chars:
                start = i
                break
        for i in reversed(range(len(text['HPC']))):
            if text['HPC'][i] not in remove_chars:
                end = i+1
                break
        new_text = {comp_id: [] for comp_id in text.keys()}
        for comp_id in new_text.keys():
            new_text[comp_id] = text[comp_id][start:end]
    elif type(text) is list:
        """字丁级文本"""
        end = len(text)
        for i in range(len(text)):
            if text[i] not in remove_chars:
                start = i
                break
        for i in reversed(range(len(text))):
            if text[i] not in remove_chars:
                end = i+1
                break
        new_text = text[start: end]
    else:
        raise TypeError
    return new_text



def num_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

