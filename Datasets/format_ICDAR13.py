import numpy as np
import os
import shutil
import tarfile, zipfile
import lmdb
import cv2
from PIL import Image
import pprint
import struct
from processHWDB import decode_HWDB_subset_v2
import pickle


def update_charset(charset, char_text):
    for char in char_text:
        if char in charset.keys():
            charset[char] += 1
        else:
            charset[char] = 1
    return charset


class Converter(object):
    def __init__(self, dict_path):
        self.chars = open(dict_path, 'r', encoding='utf-8').read().splitlines()
        self.chr2idx = {char: i for i, char in enumerate(self.chars)}

    def encode(self, str):
        return [self.chr2idx.get(char) for char in str]

    def decode(self, indices):
        chars = [self.chars[int(idx)] for idx in indices]
        return ''.join(chars)

def bytes2string(bytes, code_length, encoding):
    pos = 0
    text = ''
    while pos < len(bytes):
        code_point = bytes[pos: pos+code_length]
        if b'\xff' in code_point:
            pos += code_length # dicard the current code point
            continue
        text += str(code_point, encoding=encoding)
        pos += code_length
    return text




def decode_lmdb(env, converter, idx):
    image_key = f'image-{idx + 1:06d}'.encode()
    label_key = f'label-{idx + 1:06d}'.encode()
    num_char_key = f'numchar-{idx + 1:06d}'.encode()
    filename_key = f'filename-{idx + 1:06d}'.encode()
    image_shape_key = f'image-shape-{idx + 1:06d}'.encode()
    with env.begin(write=False) as txn:
        image_buf = txn.get(image_key)
        image_shape_buf = txn.get(image_shape_key)
        label_buf = txn.get(label_key)
        num_char_buf = txn.get(num_char_key)
        filename = txn.get(filename_key).decode()
    image_shape = np.frombuffer(image_shape_buf, np.int32).tolist()
    image = np.frombuffer(image_buf, np.uint8).reshape(*image_shape).copy()
    image = Image.fromarray(image)

    label = np.frombuffer(label_buf, np.float32).reshape(-1, 5).copy()
    label = label[:, 0].astype(np.int32)
    num_char_per_line = np.frombuffer(num_char_buf, np.int32).copy()
    line_split = np.cumsum(num_char_per_line)
    line_split = np.concatenate(([0], line_split)).astype(np.int32)
    line_labels = []
    for l_i in range(len(line_split) - 1):
        # line_labels.append(label[line_split[l_i]:line_split[l_i + 1]])
        line_labels.append(converter.decode(label[line_split[l_i]:line_split[l_i + 1]]))
    return image, line_labels, filename

def decode_dgrl(file_path, log_txt):
    with open(file_path, 'rb') as f:
        # content = f.read()
        """Header"""
        header_size = struct.unpack('i', f.read(4))[0]
        format_code = str(f.read(8), 'ascii').rstrip('\x00')
        len_illus = header_size-36
        illustration = str(f.read(len_illus), 'ascii').rstrip('\x00')
        code_type = str(f.read(20), 'ascii').rstrip('\x00')
        code_length = struct.unpack('h', f.read(2))[0]
        bits_per_pixel = struct.unpack('h', f.read(2))[0]
        bytes_per_pixel = bits_per_pixel // 8
        """Image records"""
        height = struct.unpack('i', f.read(4))[0]
        width = struct.unpack('i', f.read(4))[0]
        num_lines = struct.unpack('i', f.read(4))[0]
        image = np.full((height, width), fill_value=255, dtype=np.uint8)
        """Line records"""
        line_texts = []
        line_bboxs = []
        left, top = 99999, 99999
        right, bottom = -99999, -99999
        for i in range(num_lines):
            num_chars = struct.unpack('i', f.read(4))[0]
            label = f.read(num_chars*code_length)
            # label = str(label.replace(b'\xFF', b''), 'gbk').replace('\x00', '') # \xFF for garbage
            assert code_length == 2
            label = bytes2string(label, code_length, 'gb18030').replace('\x00', '')
            line_texts.append(label)
            # if len(label) != num_chars:
            #     print(f'len(label)!=num_chars in {i+1}th line of {file_path}')
            y = struct.unpack('i', f.read(4))[0]
            x = struct.unpack('i', f.read(4))[0]
            h = struct.unpack('i', f.read(4))[0]
            w = struct.unpack('i', f.read(4))[0]

            if y < 0:
                log_txt += f'y = {y} in {i + 1}th line of {file_path}!\n'
                y = 0
            if x < 0:
                log_txt += f'x = {x} in {i + 1}th line of {file_path}!\n'
                x = 0
            assert h > 0
            assert w > 0

            left = min(left, x)
            top = min(top, y)
            right = max(right, x+w)
            bottom = max(bottom, y+h)
            line_bboxs.append((y, x, h, w))
            assert bytes_per_pixel == 1
            line_image = np.frombuffer(f.read(h*w*bytes_per_pixel), np.uint8).reshape(h, w).copy()
            # line_image = 255 - line_image
            image[y:y+h, x:x+w][line_image!=255] = line_image[line_image!=255]
        image = Image.fromarray(image[top:bottom, left:right])
        line_bboxs = [(y-top, x-left, h, w) for y, x, h, w in line_bboxs]
        return image, line_texts, line_bboxs, log_txt


def format_ICDAR13_paragraph(datasets):
    target_root = "formatted/ICDAR13_paragraph"
    if os.path.exists(target_root):
        shutil.rmtree(target_root)
    os.makedirs(target_root, exist_ok=True)
    converter = Converter('casia-hwdb.txt')
    gt = dict()
    charset = dict()
    charset_split = dict()
    log_txt = ''
    for fn, set_name, type in datasets:
        if not os.path.exists(fn):
            print("error - {} not found".format(fn))
        assert set_name in ['train', 'valid', 'test']
        name = os.path.splitext(os.path.basename(fn))[0]
        target_folder = os.path.join(target_root, set_name, name)
        os.makedirs(target_folder)
        if set_name not in gt.keys():
            gt[set_name] = dict()
        if set_name not in charset_split.keys():
            charset_split[set_name] = dict()
        if type == 'zip':
            with zipfile.ZipFile(fn, 'r') as zip_ref:
                zip_ref.extractall(target_folder)
            to_del = []
            # decode_HWDB_subset_v2(target_folder, target_folder, target_folder)
            for filename in os.listdir(target_folder):
                image, line_labels, line_bboxs, log_txt = decode_dgrl(os.path.join(target_folder, filename), log_txt)
                id = os.path.splitext(filename)[0]
                image.save(os.path.join(target_folder, id + '.png'))
                text = '\n'.join(line_labels)
                with open(os.path.join(target_folder, id + '.txt'), 'w', encoding='utf-8') as f:
                    f.write(text)
                lines = [{'text': lt,
                          'left': x,
                          'right': x+w,
                          'top': y,
                          'bottom': y+h} for lt, (y, x, h, w) in zip(line_labels, line_bboxs)]
                gt[set_name][name + '/' + id + '.png'] = {
                    'text': text,
                    'lines': lines
                }
                charset = update_charset(charset, text)
                charset_split[set_name] = update_charset(charset_split[set_name], text)
                to_del.append(os.path.join(target_folder, filename))

            for path_to_del in to_del:
                os.remove(path_to_del)


        elif type == 'lmdb':
            env = lmdb.open(fn, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
            with env.begin(write=False) as txn:
                num_samples = int(txn.get('num-samples'.encode()).decode())
            for i in range(num_samples):
                image, line_labels, id = decode_lmdb(env, converter, i)
                image.save(os.path.join(target_folder, id + '.png'))
                text = '\n'.join(line_labels)
                with open(os.path.join(target_folder, id + '.txt'), 'w', encoding='utf-8') as f:
                    f.write(text)
                lines = [{'text': line_labels[i]} for i in range(len(line_labels))]
                gt[set_name][name + '/' + id + '.png'] = {
                    'text': text,
                    'lines': lines
                }
                charset = update_charset(charset, text)
                charset_split[set_name] = update_charset(charset_split[set_name], text)
        else:
            raise NotImplementedError
    for set_name in charset_split.keys():
        charset_split[set_name] = dict(sorted(charset_split[set_name].items()))
    charset = dict(sorted(charset.items()))
    with open(os.path.join(target_root, "labels.pkl"), "wb") as f:
        pickle.dump({
            "ground_truth": gt,
            "charset": charset,
            "charset_split": charset_split
        }, f)

    """输出charset到txt文件中"""
    with open(os.path.join(target_root, "charset.txt"), "w", encoding='utf-8') as f:
        txt = 'index\tcharacter\tunicode\tcount\n'
        for i, (char, cnt) in enumerate(charset.items()):
            hex_txt = '+'.join([hex(ord(letter)) for letter in char])
            char = char if char != '\n' else '\\n'
            txt += f'{i}\t{char}\t{hex_txt}\t{cnt}\n'
        f.write(txt)
    with open(os.path.join(target_root, 'log.txt'), 'w', encoding='utf-8') as f:
        f.write(log_txt)



datasets = [
    ('c:/myDatasets/VAN/raw/CASIA_HWDB2.0-2.2/HWDB2.0Train.zip', 'train', 'zip'),
    ('c:/myDatasets/VAN/raw/CASIA_HWDB2.0-2.2/HWDB2.1Train.zip', 'train', 'zip'),
    ('c:/myDatasets/VAN/raw/CASIA_HWDB2.0-2.2/HWDB2.2Train.zip', 'train', 'zip'),
    ('c:/myDatasets/VAN/raw/CASIA_HWDB2.0-2.2/HWDB2.0Test.zip', 'train', 'zip'),
    ('c:/myDatasets/VAN/raw/CASIA_HWDB2.0-2.2/HWDB2.1Test.zip', 'train', 'zip'),
    ('c:/myDatasets/VAN/raw/CASIA_HWDB2.0-2.2/HWDB2.2Test.zip', 'train', 'zip'),
    ('c:/myDatasets/VAN/raw/IC13Comp', 'valid', 'lmdb')
]

test_sets = {
    'c:/myDatasets/VAN/raw/IC13Comp'
}

source_folder = "c:/myDatasets/VAN/raw/ICDAR13"


if __name__ == '__main__':
    format_ICDAR13_paragraph(datasets)
    pass