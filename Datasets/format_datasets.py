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
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
import tarfile, zipfile
import pickle
import numpy as np
import torch
from PIL import Image
import cv2
from myUtils.Tibetan import string2char_list


def format_IAM_line():
    """
    Format the IAM dataset at line level with the commonly used split (6,482 for train, 976 for validation and 2,915 for test)
    """
    source_folder = "raw/IAM"
    target_folder = "formatted/IAM_lines"
    tar_filename = "lines.tgz"
    line_folder_path = os.path.join(target_folder, "lines")

    tar_path = os.path.join(source_folder, tar_filename)
    if not os.path.isfile(tar_path):
        print("error - {} not found".format(tar_path))
        exit(-1)

    os.makedirs(target_folder, exist_ok=True)
    tar = tarfile.open(tar_path)
    tar.extractall(line_folder_path)
    tar.close()

    set_names = ["train", "valid", "test"]
    gt = {
        "train": dict(),
        "valid": dict(),
        "test": dict()
    }
    charset = set()

    for set_name in set_names:
        id = 0
        current_folder = os.path.join(target_folder, set_name)
        os.makedirs(current_folder, exist_ok=True)
        xml_path = os.path.join(source_folder, "{}.xml".format(set_name))
        xml_root = ET.parse(xml_path).getroot()
        for page in xml_root:
            name = page.attrib.get("FileName").split("/")[-1].split(".")[0]
            img_fold_path = os.path.join(line_folder_path, name.split("-")[0], name)
            img_paths = [os.path.join(img_fold_path, p) for p in sorted(os.listdir(img_fold_path))]
            for i, line in enumerate(page[2]):
                label = line.attrib.get("Value")
                img_name = "{}_{}.png".format(set_name, id)
                gt[set_name][img_name] = {
                    "text": label,
                }
                charset = charset.union(set(label))
                new_path = os.path.join(current_folder, img_name)
                os.replace(img_paths[i], new_path)
                id += 1

    shutil.rmtree(line_folder_path)
    with open(os.path.join(target_folder, "labels.pkl"), "wb") as f:
         pickle.dump({
             "ground_truth": gt,
             "charset": sorted(list(charset)),
         }, f)


def format_IAM_paragraph():
    """
    Format the IAM dataset at paragraph level with the commonly used split (747 for train, 116 for validation and 336 for test)
    """
    source_folder = "raw/IAM"
    target_folder = "formatted/IAM_paragraph"
    img_folder_path = os.path.join(target_folder, "images")

    os.makedirs(target_folder, exist_ok=True)

    tar_filenames = ["formsA-D.tgz", "formsE-H.tgz", "formsI-Z.tgz"]
    tar_paths = [os.path.join(source_folder, name) for name in tar_filenames]
    for tar_path in tar_paths:
        if not os.path.isfile(tar_path):
            print("error - {} not found".format(tar_path))
            exit(-1)
        tar = tarfile.open(tar_path)
        tar.extractall(img_folder_path)
        tar.close()

    gt = {
        "train": dict(),
        "valid": dict(),
        "test": dict()
    }
    charset = set()

    for set_name in ["train", "valid", "test"]:
        new_folder = os.path.join(target_folder, set_name)
        os.makedirs(new_folder, exist_ok=True)
        xml_path = os.path.join(source_folder, "{}.xml".format(set_name))
        xml_root = ET.parse(xml_path).getroot()
        for page in xml_root:
            name = page.attrib.get("FileName").split("/")[-1].split(".")[0]
            img_path = os.path.join(img_folder_path, name + ".png")
            new_name = "{}_{}.png".format(set_name, len(os.listdir(new_folder)))
            new_img_path = os.path.join(new_folder, new_name)
            lines = []
            full_text = ""
            for section in page:
                if section.tag != "Paragraph":
                    continue
                p_left, p_right = int(section.attrib.get("Left")), int(section.attrib.get("Right"))
                p_bottom, p_top = int(section.attrib.get("Bottom")), int(section.attrib.get("Top"))
                for i, line in enumerate(section):
                    words = []
                    for word in line:
                        words.append({
                            "text": word.attrib.get("Value"),
                            "left": int(word.attrib.get("Left")) - p_left,
                            "bottom": int(word.attrib.get("Bottom")) - p_top,
                            "right": int(word.attrib.get("Right")) - p_left,
                            "top": int(word.attrib.get("Top")) - p_top
                        })
                    lines.append({
                        "text": line.attrib.get("Value"),
                        "left": int(line.attrib.get("Left")) - p_left,
                        "bottom": int(line.attrib.get("Bottom")) - p_top,
                        "right": int(line.attrib.get("Right")) - p_left,
                        "top": int(line.attrib.get("Top")) - p_top,
                        "words": words
                    })
                    full_text = "{}{}\n".format(full_text, lines[-1]["text"])
                paragraph = {
                    "text": full_text[:-1],
                    "lines": lines
                }
                gt[set_name][new_name] = paragraph
                charset = charset.union(set(full_text))

                with Image.open(img_path) as pil_img:
                    img = np.array(pil_img)
                img = img[p_top:p_bottom + 1, p_left:p_right + 1]
                Image.fromarray(img).save(new_img_path)

    shutil.rmtree(img_folder_path)
    with open(os.path.join(target_folder, "labels.pkl"), "wb") as f:
         pickle.dump({
             "ground_truth": gt,
             "charset": sorted(list(charset)),
         }, f)


def format_RIMES_line():
    """
    Format the RIMES dataset at line level with the commonly used split (the last 100 paragraph training samples are taken for the valid set)
    resulting in: 10,532 for training, 801 for validation and 778 for test.
    """
    source_folder = "raw/RIMES"
    target_folder = "formatted/RIMES_lines"
    img_folder_path = os.path.join(target_folder, "images_gray")
    os.makedirs(target_folder, exist_ok=True)

    eval_xml_path = os.path.join(source_folder, "eval_2011_annotated.xml")
    train_xml_path = os.path.join(source_folder, "training_2011.xml")
    for label_path in [eval_xml_path, train_xml_path]:
        if not os.path.isfile(label_path):
            print("error - {} not found".format(label_path))
            exit(-1)

    tar_filenames = ["eval_2011_gray.tar", "training_2011_gray.tar"]
    tar_paths = [os.path.join(source_folder, name) for name in tar_filenames]
    for tar_path in tar_paths:
        if not os.path.isfile(tar_path):
            print("error - {} not found".format(tar_path))
            exit(-1)
        tar = tarfile.open(tar_path)
        tar.extractall(target_folder)
        tar.close()

    nb_training_samples = sum([1 if "train" in name else 0 for name in os.listdir(img_folder_path)])
    nb_valid_samples = 100
    begin_valid_ind = nb_training_samples - nb_valid_samples

    gt = {
        "train": dict(),
        "valid": dict(),
        "test": dict()
    }

    for set_name in gt.keys():
        os.makedirs(os.path.join(target_folder, set_name), exist_ok=True)

    charset = set()
    for set_name, xml_path in zip(["train", "eval"], [train_xml_path, eval_xml_path]):
        xml_root = ET.parse(xml_path).getroot()
        for page in xml_root:
            name = page.attrib.get("FileName").split("/")[-1].split(".")[0]
            img_path = os.path.join(img_folder_path, name+".png")
            img = np.array(Image.open(img_path, "r"))
            if set_name == "train":
                new_set_name = "train" if int(name.split("-")[-1])<begin_valid_ind else "valid"
            else:
                new_set_name = "test"
            fold = os.path.join(target_folder, new_set_name)
            for i, line in enumerate(page[0]):
                label = line.attrib.get("Value")
                new_name = "{}_{}.png".format(new_set_name, len(os.listdir(fold)))
                new_path = os.path.join(fold, new_name)
                left = max(0, int(line.attrib.get("Left")))
                bottom = int(line.attrib.get("Bottom"))
                right = int(line.attrib.get("Right"))
                top = max(0, int(line.attrib.get("Top")))
                line_img = img[top:bottom, left:right]
                try:
                    Image.fromarray(line_img).save(new_path)
                except:
                    print("problem")
                paragraph = {
                    "text": label,
                }
                charset = charset.union(set(label))
                gt[new_set_name][new_name] = paragraph

    with open(os.path.join(target_folder, "labels.pkl"), "wb") as f:
         pickle.dump({
             "ground_truth": gt,
             "charset": sorted(list(charset)),
         }, f)

    shutil.rmtree(img_folder_path)


def format_RIMES_paragraph():
    """
    Format the RIMES dataset at paragraph level with the official split (the last 100 training samples are taken for the valid set)
    resulting in 1,400 for training, 100 for validation and 100 for test
    """
    source_folder = "raw/RIMES"
    target_folder = "formatted/RIMES_paragraph"
    img_folder_path = os.path.join(target_folder, "images_gray")
    os.makedirs(target_folder, exist_ok=True)

    eval_xml_path = os.path.join(source_folder, "eval_2011_annotated.xml")
    train_xml_path = os.path.join(source_folder, "training_2011.xml")
    for label_path in [eval_xml_path, train_xml_path]:
        if not os.path.isfile(label_path):
            print("error - {} not found".format(label_path))
            exit(-1)

    tar_filenames = ["eval_2011_gray.tar", "training_2011_gray.tar"]
    tar_paths = [os.path.join(source_folder, name) for name in tar_filenames]
    for tar_path in tar_paths:
        if not os.path.isfile(tar_path):
            print("error - {} not found".format(tar_path))
            exit(-1)
        tar = tarfile.open(tar_path)
        tar.extractall(target_folder)
        tar.close()

    nb_training_samples = sum([1 if "train" in name else 0 for name in os.listdir(img_folder_path)])
    nb_valid_samples = 100
    begin_valid_ind = nb_training_samples - nb_valid_samples

    gt = {
        "train": dict(),
        "valid": dict(),
        "test": dict()
    }

    for set_name in gt.keys():
        os.makedirs(os.path.join(target_folder, set_name), exist_ok=True)

    charset = set()
    for set_name, xml_path in zip(["train", "eval"], [train_xml_path, eval_xml_path]):
        xml_root = ET.parse(xml_path).getroot()
        for page in xml_root:
            name = page.attrib.get("FileName").split("/")[-1].split(".")[0]
            img_path = os.path.join(img_folder_path, name+".png")
            if set_name == "train":
                new_set_name = "train" if int(name.split("-")[-1])<begin_valid_ind else "valid"
            else:
                new_set_name = "test"
            fold = os.path.join(target_folder, new_set_name)
            new_name = "{}_{}.png".format(new_set_name, len(os.listdir(fold)))
            new_path = os.path.join(fold, new_name)
            os.replace(img_path, new_path)
            lines = []
            full_text = ""
            for i, line in enumerate(page[0]):

                lines.append({
                    "text": line.attrib.get("Value"),
                    "left": int(line.attrib.get("Left")),
                    "bottom": int(line.attrib.get("Bottom")),
                    "right": int(line.attrib.get("Right")),
                    "top": int(line.attrib.get("Top")),
                })
                full_text = "{}{}\n".format(full_text, lines[-1]["text"])
            paragraph = {
                "text": full_text[:-1],
                "lines": lines
            }
            charset = charset.union(set(full_text))
            gt[new_set_name][new_name] = paragraph

    with open(os.path.join(target_folder, "labels.pkl"), "wb") as f:
         pickle.dump({
             "ground_truth": gt,
             "charset": sorted(list(charset)),
         }, f)

    shutil.rmtree(img_folder_path)


def format_READ2016_line():
    """
    Format the READ 2016 dataset at line level with the official split (8,349 for training, 1,040 for validation and 1,138 for test)
    """
    source_folder = "raw/READ_2016"
    target_folder = "formatted/READ_2016_lines"
    if os.path.isdir(target_folder):
        shutil.rmtree(target_folder)
    os.makedirs(target_folder)

    tar_filenames = ["Test-ICFHR-2016.tgz", "Train-And-Val-ICFHR-2016.tgz"]
    tar_paths = [os.path.join(source_folder, name) for name in tar_filenames]
    for tar_path in tar_paths:
        if not os.path.isfile(tar_path):
            print("error - {} not found".format(tar_path))
            exit(-1)
        tar = tarfile.open(tar_path)
        tar.extractall(target_folder)
        tar.close()

    os.rename(os.path.join(target_folder, "PublicData", "Training"), os.path.join(target_folder, "train"))
    os.rename(os.path.join(target_folder, "PublicData", "Validation"), os.path.join(target_folder, "valid"))
    os.rename(os.path.join(target_folder, "Test-ICFHR-2016"), os.path.join(target_folder, "test"))
    os.rmdir(os.path.join(target_folder, "PublicData"))
    for set_name in ["train", "valid", ]:
        for filename in os.listdir(os.path.join(target_folder, set_name, "Images")):
            filepath = os.path.join(target_folder, set_name, "Images", filename)
            if os.path.isfile(filepath):
                os.rename(filepath, os.path.join(target_folder, set_name, filename))
        os.rmdir(os.path.join(target_folder, set_name, "Images"))

    gt = {
        "train": dict(),
        "valid": dict(),
        "test": dict()
    }

    charset = set()
    for set_name in ["train", "valid", "test"]:
        img_fold_path = os.path.join(target_folder, set_name)
        xml_fold_path = os.path.join(target_folder, set_name, "page")
        i = 0
        for xml_file_name in sorted(os.listdir(xml_fold_path)):
            if xml_file_name.split(".")[-1] != "xml":
                continue
            filename = xml_file_name.split(".")[0]
            img_path = os.path.join(img_fold_path, filename+".JPG")
            xml_file_path = os.path.join(xml_fold_path, xml_file_name)
            xml_root = ET.parse(xml_file_path).getroot()
            img = np.array(Image.open(img_path))
            for text_region in xml_root[1][1:]:
                if text_region.tag.split("}")[-1] != "TextRegion":
                    continue
                for balise in text_region:
                    if balise.tag.split("}")[-1] != "TextLine":
                        continue
                    for sub in balise:
                        if sub.tag.split("}")[-1] == "Coords":
                            points = sub.attrib["points"].split(" ")
                            x_points, y_points = list(), list()
                            for p in points:
                                y_points.append(int(p.split(",")[1]))
                                x_points.append(int(p.split(",")[0]))
                        elif sub.tag.split("}")[-1] == "TextEquiv":
                            line_label = sub[0].text
                    if line_label is None:
                        continue
                    top, bottom, left, right = np.min(y_points), np.max(y_points), np.min(x_points), np.max(x_points)
                    new_img_name = "{}_{}.jpeg".format(set_name, i)
                    new_img_path = os.path.join(img_fold_path, new_img_name)
                    curr_img = img[top:bottom + 1, left:right + 1]
                    Image.fromarray(curr_img).save(new_img_path)
                    gt[set_name][new_img_name] = {"text": line_label, }
                    charset = charset.union(line_label)
                    i += 1
                    line_label = None
            os.remove(img_path)
        shutil.rmtree(xml_fold_path)

    with open(os.path.join(target_folder, "labels.pkl"), "wb") as f:
        pickle.dump({
            "ground_truth": gt,
            "charset": sorted(list(charset)),
        }, f)


def format_READ2016_paragraph():
    """
    Format the READ 2016 dataset at paragraph level with the official split (1,584 for training 179, for validation and 197 for test)
    """
    source_folder = "raw/READ_2016"
    target_folder = "formatted/READ_2016_paragraph"
    if os.path.isdir(target_folder):
        shutil.rmtree(target_folder)
    os.makedirs(target_folder)

    tar_filenames = ["Test-ICFHR-2016.tgz", "Train-And-Val-ICFHR-2016.tgz"]
    tar_paths = [os.path.join(source_folder, name) for name in tar_filenames]
    for tar_path in tar_paths:
        if not os.path.isfile(tar_path):
            print("error - {} not found".format(tar_path))
            exit(-1)
        tar = tarfile.open(tar_path)
        tar.extractall(target_folder)
        tar.close()

    os.rename(os.path.join(target_folder, "PublicData", "Training"), os.path.join(target_folder, "train"))
    os.rename(os.path.join(target_folder, "PublicData", "Validation"), os.path.join(target_folder, "valid"))
    os.rename(os.path.join(target_folder, "Test-ICFHR-2016"), os.path.join(target_folder, "test"))
    os.rmdir(os.path.join(target_folder, "PublicData"))
    for set_name in ["train", "valid", ]:
        for filename in os.listdir(os.path.join(target_folder, set_name, "Images")):
            filepath = os.path.join(target_folder, set_name, "Images", filename)
            if os.path.isfile(filepath):
                os.rename(filepath, os.path.join(target_folder, set_name, filename))
        os.rmdir(os.path.join(target_folder, set_name, "Images"))

    gt = {
        "train": dict(),
        "valid": dict(),
        "test": dict()
    }

    charset = set()
    for set_name in ["train", "valid", "test"]:
        img_fold_path = os.path.join(target_folder, set_name)
        xml_fold_path = os.path.join(target_folder, set_name, "page")
        i = 0
        for xml_file_name in os.listdir(xml_fold_path):
            if xml_file_name.split(".")[-1] != "xml":
                continue
            filename = xml_file_name.split(".")[0]
            img_path = os.path.join(img_fold_path, filename+".JPG")
            xml_file_path = os.path.join(xml_fold_path, xml_file_name)
            xml_root = ET.parse(xml_file_path).getroot()
            img = np.array(Image.open(img_path))
            for text_region in xml_root[1][1:]:
                if text_region.tag.split("}")[-1] != "TextRegion":
                    continue
                for balise in text_region:
                    if balise.tag.split("}")[-1] == "Coords":
                        points = balise.attrib["points"].split(" ")
                        x_points, y_points = list(), list()
                        for p in points:
                            y_points.append(int(p.split(",")[1]))
                            x_points.append(int(p.split(",")[0]))
                    if balise.tag.split("}")[-1] == "TextEquiv":
                        pg_label = balise[0].text
                if pg_label is None:
                    continue
                top, bottom, left, right = np.min(y_points), np.max(y_points), np.min(x_points), np.max(x_points)
                new_img_name = "{}_{}.jpeg".format(set_name, i)
                new_img_path = os.path.join(img_fold_path, new_img_name)
                curr_img = img[top:bottom + 1, left:right + 1]
                Image.fromarray(curr_img).save(new_img_path)
                gt[set_name][new_img_name] = {"text": pg_label, }
                charset = charset.union(pg_label)
                i += 1
                pg_label = None
            os.remove(img_path)
        shutil.rmtree(xml_fold_path)

    with open(os.path.join(target_folder, "labels.pkl"), "wb") as f:
        pickle.dump({
            "ground_truth": gt,
            "charset": sorted(list(charset)),
        }, f)

def get_pn_split(seed):
    line_annot_file = 'c:/myDatasets/bj_Ganjur/line_annot.json'

    with open(line_annot_file, 'r') as f:
        line_annot = json.load(f)

    # page name list
    pn_list = list()
    for pn, annot in line_annot.items():
        if 'content' in annot[0] and annot[0]['content'] != '':
            pn_list.append(pn)
    pn_list.sort()

    indices = list(range(0, len(pn_list)))
    num_train = int(len(pn_list) * 0.8)
    random.seed(seed)
    indices_train = random.sample(indices, num_train)
    indices_train.sort()
    pn_train = [pn_list[i] for i in indices_train]
    pn_val = [pn_list[i] for i in indices if i not in indices_train]
    return pn_train, pn_val

data_root = 'c:/myDatasets/bj_Ganjur'


def update_charset(charset, char_list):
    for char in char_list:
        if char in charset.keys():
            charset[char] += 1
        else:
            charset[char] = 1
    return charset

def format_BJK_212_line(seed = 1, image_type='bin', fine=True):
    line_annot_file = os.path.join(data_root, 'line_annot.json')
    if image_type == 'color':
        img_root = os.path.join(data_root, 'orig_Image')
    elif image_type == 'bin':
        img_root = os.path.join(data_root, '212_bin_bitwise_res')
    else:
        raise ValueError

    with open(line_annot_file, 'r') as f:
        line_annot = json.load(f)

    pn_tr, pn_val = get_pn_split(seed=seed)

    gt = {
        "train": dict(),
        "valid": dict()
    }
    charset = dict()
    charset_split = dict()

    pn_dict = {
        'train': pn_tr,
        'valid': pn_val
    }

    target_folder = f"formatted/BJK_212{f'_{image_type}' if image_type != 'color' else ''}_lines"

    os.makedirs(target_folder, exist_ok=True)
    shutil.rmtree(target_folder)

    for set_name, pn_list in pn_dict.items():
        charset_split[set_name] = dict()
        os.makedirs(os.path.join(target_folder, set_name), exist_ok=True)
        for pn in pn_list:
            name, ext = os.path.splitext(pn)
            if image_type == 'bin':
                img = cv2.imread(os.path.join(img_root, pn), cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(os.path.join(img_root, pn))
            if img is None:
                continue
            for li, line in enumerate(line_annot[pn]):
                char_list = string2char_list(line['content'])
                line_name = name + '_' + str(li+1) + ext
                gt[set_name][line_name] = {'text': line['content'], 'char_list': char_list}
                # charset = charset.union(set(char_list))
                charset = update_charset(charset, char_list)
                # charset_split[set_name] = charset_split[set_name].union(set(char_list))
                charset_split[set_name] = update_charset(charset_split[set_name], char_list)
                if fine:
                    # 通过contour分割出文本行图像
                    mask = np.zeros_like(img)
                    fill_color = (1, 1, 1) if len(mask.shape) == 3 else 1
                    points = np.array(line['contour'])
                    cv2.fillPoly(mask, pts=[points], color=fill_color)
                    inverse_mask = np.ones_like(img) - mask
                    # 白色背景
                    canvas = np.full_like(img, 255)
                    masked = img * mask + canvas * inverse_mask
                    x, y, w, h = line['bbox']
                    if image_type == 'bin':
                        line_img = masked[y:y + h, x:x + w]
                    else:
                        line_img = masked[y:y + h, x:x + w, :]
                else:
                    # 通过bbox分割出文本行图像
                    x, y, w, h = line['bbox']
                    line_img = img[y:y + h, x:x + w, :]

                fn = os.path.join(target_folder, set_name, line_name)
                cv2.imwrite(fn, line_img)
        charset_split[set_name] = dict(sorted(charset_split[set_name].items()))
    charset = dict(sorted(charset.items()))

    with open(os.path.join(target_folder, "labels.pkl"), "wb") as f:
        pickle.dump({
            "ground_truth": gt,
            "charset": charset,
            'charset_split': charset_split
        }, f)

    """输出charset到txt文件中"""
    with open(os.path.join(target_folder, "charset.txt"), "w", encoding='utf-8') as f:
        txt = 'index\tcharacter\tunicode\tcount\n'
        for i, (char, cnt) in enumerate(charset.items()):
            hex_txt = '+'.join([hex(ord(letter)) for letter in char])
            char = char if char != '\n' else '\\n'
            txt += f'{i}\t{char}\t{hex_txt}\t{cnt}\n'
        f.write(txt)

    # intersection of all charsets
    intr = set()
    # union of all charsets
    un = set()
    for idx, charset in enumerate(charset_split.values()):
        if idx == 0:
            intr = set(charset.keys())
            un = set(charset.keys())
        else:
            intr = intr.intersection(set(charset.keys()))
            un = un.union(set(charset.keys()))

    ratio = len(intr) / len(un)
    print('line_stats:')
    print(f'intersection ratio: {ratio:.4f}')




def format_BJK_212_paragraph(seed = 1, image_type='bin'):

    if image_type == 'color':
        img_root = os.path.join(data_root, 'orig_Image')
    elif image_type == 'bin':
        img_root = os.path.join(data_root, '212_bin_bitwise_res')
    else:
        raise ValueError

    line_annot_file = os.path.join(data_root, 'line_annot.json')
    with open(line_annot_file, 'r') as f:
        line_annot = json.load(f)

    pn_tr, pn_val = get_pn_split(seed=seed)

    pn_dict = {
        'train': pn_tr,
        'valid': pn_val
    }

    gt = dict()

    charset = dict()
    charset_split = dict()

    # target_folder = "formatted/BJK_212_185_paragraph"
    target_folder = f"formatted/BJK_212{f'_{image_type}' if image_type != 'color' else ''}_paragraph"
    os.makedirs(target_folder, exist_ok=True)
    shutil.rmtree(target_folder)

    for set_name, pn_list in pn_dict.items():
        gt[set_name] = dict()
        charset_split[set_name] = dict()
        target_img_folder = os.path.join(target_folder, set_name)
        os.makedirs(target_img_folder, exist_ok=True)
        for pn in pn_list:

            if image_type == 'bin':
                img = cv2.imread(os.path.join(img_root, pn), cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(os.path.join(img_root, pn))

            if img is None:
                continue

            gt[set_name][pn] = {
                'text': '',
                'char_list': [],
                'lines': []
            }

            left = 999999
            right = -999999
            top = 999999
            bottom = -999999

            for line_idx, line in enumerate(line_annot[pn]):
                assert '\n' not in line['content']
                gt[set_name][pn]['text'] += line['content']
                # assert line['content'][0] != ' ' and line['content'][-1] != ' '
                # if pn in ['002.29.png'] and line_idx == 1:  # debug
                #     print('pass')
                char_list = string2char_list(line['content'])
                gt[set_name][pn]['char_list'] += char_list
                if line_idx < len(line_annot[pn]) - 1:
                    """不是最后一行"""
                    gt[set_name][pn]['text'] += '\n'
                    gt[set_name][pn]['char_list'] += ['\n']
                    charset = update_charset(charset, '\n')
                    charset_split[set_name] = update_charset(charset_split[set_name], '\n')

                # charset = charset.union(set(char_list))
                charset = update_charset(charset, char_list)
                # charset_split[set_name] = charset_split[set_name].union(set(char_list))
                charset_split[set_name] = update_charset(charset_split[set_name], char_list)

                x, y, w, h = line['bbox']

                gt[set_name][pn]['lines'].append({
                    'text': line['content'],
                    'char_list': char_list,
                    'left': x,
                    'right': x+w,
                    'top': y,
                    'bottom': y+h,
                    'words': None
                })

                left = min(left, x)
                right = max(right, x+w)
                top = min(top, y)
                bottom = max(bottom, y+h)
            assert left <= right and top <= bottom

            if image_type == 'bin':
                paragraph_img = img[top:bottom, left:right]
            else:
                paragraph_img = img[top:bottom, left:right, :]

            target_img_path = os.path.join(target_img_folder, pn)
            cv2.imwrite(target_img_path, paragraph_img)

        charset_split[set_name] = dict(sorted(charset_split[set_name].items()))

    charset = dict(sorted(charset.items()))

    with open(os.path.join(target_folder, "labels.pkl"), "wb") as f:
        pickle.dump({
            "ground_truth": gt,
            "charset": charset,
            "charset_split": charset_split
        }, f)

    """输出charset到txt文件中"""
    with open(os.path.join(target_folder, "charset.txt"), "w", encoding='utf-8') as f:
        txt = 'index\tcharacter\tunicode\tcount\n'
        for i, (char, cnt) in enumerate(charset.items()):
            hex_txt = '+'.join([hex(ord(letter)) for letter in char])
            char = char if char != '\n' else '\\n'
            txt += f'{i}\t{char}\t{hex_txt}\t{cnt}\n'
        f.write(txt)

    # intersection of all charsets
    intr = set()
    # union of all charsets
    un = set()
    for idx, charset in enumerate(charset_split.values()):
        if idx == 0:
            intr = set(charset.keys())
            un = set(charset.keys())
        else:
            intr = intr.intersection(set(charset.keys()))
            un = un.union(set(charset.keys()))

    ratio = len(intr) / len(un)
    print('paragraph_stats:')
    print(f'intersection ratio: {ratio:.4f}')




if __name__ == "__main__":
    format_IAM_line()
    format_IAM_paragraph()

    # format_RIMES_line()
    # format_RIMES_paragraph()

    # format_READ2016_line()
    # format_READ2016_paragraph()

