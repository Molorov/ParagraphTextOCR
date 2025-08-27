import numpy as np
import edit_distance as ed
import editdistance


def match_lines(line_labels, line_results, enforce_order=False, log=True):
    to_log = ''

    De = 0
    Se = 0
    Ie = 0
    Len = 0.0


    AR_mt = np.zeros((len(line_labels), len(line_results)))
    error_mt = np.zeros((len(line_labels), len(line_results), 3))
    for i, line_label in enumerate(line_labels):
        for j, line_result in enumerate(line_results):
            dis, error, _, _, _ = cal_distance(line_label, line_result)
            assert dis == editdistance.eval(line_label, line_result)
            AR_mt[i, j] = 1 - float(dis) / len(line_label)
            error_mt[i, j] = error

    Score_mt = np.zeros((len(line_labels), len(line_results)))
    if enforce_order:
        pos_i = np.expand_dims(np.arange(len(line_labels)), axis=1)
        pos_i = np.repeat(pos_i, len(line_results), axis=1)
        pos_i = pos_i / len(line_labels)
        pos_j = np.expand_dims(np.arange(len(line_results)), axis=0)
        pos_j = np.repeat(pos_j, len(line_labels), axis=0)
        pos_j = pos_j / len(line_results)
        pos_shift = (1.-np.abs(pos_i-pos_j))
        Score_mt = AR_mt * pos_shift
    else:
        Score_mt = AR_mt

    matched_indices = []
    label_matched = np.zeros(len(line_labels), dtype=np.int32)
    pred_matched = np.zeros(len(line_results), dtype=np.int32)
    sorted_indies = np.argsort(Score_mt.flatten())[::-1]
    for index in sorted_indies:
        if (label_matched == 1).all() or (pred_matched == 1).all():
            break
        i = index // len(line_results)
        j = index % len(line_results)
        if label_matched[i] == 0 and pred_matched[j] == 0 and len(line_results[j]):
            label_matched[i] = 1
            pred_matched[j] = 1
            matched_indices.append((i, j))
            error = error_mt[i, j]
            De += error[0]
            Se += error[1]
            Ie += error[2]
            Len += len(line_labels[i])


    matched_indices = sorted(matched_indices, key=lambda tup: tup[0])
    matched_lines = []
    for l in range(len(matched_indices)):
        # if l > 0 and matched_indices[l][1] <= matched_indices[l-1][1]:
        #     """保证 i 和 j 的单调性（增加）"""
        #     to_log += f'discard the tuple ({i}， {j})\n'
        #     continue
        matched_lines.append(line_results[matched_indices[l][1]])

    # if len(matched_lines) != len(line_labels):
    #     print('debug')

    return matched_lines, De, Se, Ie, Len, to_log


def cal_distance(label_list, pre_list):
    y = ed.SequenceMatcher(a=label_list, b=pre_list)
    yy = y.get_opcodes()
    insert = 0
    delete = 0
    replace = 0
    label_index = []
    pre_index = []
    consec_eql = []
    # import pdb;pdb.set_trace()
    for i, item in enumerate(yy):
        if item[0] == 'insert':
            insert += item[-1] - item[-2]
        if item[0] == 'delete':
            delete += item[2] - item[1]
        if item[0] == 'replace':
            replace += item[-1] - item[-2]
        if item[0] == 'equal':
            label_index.append(item[1])
            pre_index.append(item[3])
            if i != (len(yy) - 1):
                if yy[i + 1][0] == 'equal':
                    consec_eql.append(item[3])
            else:
                consec_eql.append(item[3])
    distance = insert + delete + replace
    return distance, (delete, replace, insert), label_index, pre_index, consec_eql