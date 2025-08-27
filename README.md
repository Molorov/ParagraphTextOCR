# Composite-CTC: a fully convolutional network for end-to-end paragraph handwritten text recognition
This repository is a public implementation of the paper: "Composite-CTC for Full-page Handwritten Text Recognition: Non-autoregressive, Minimal Supervision, and Lightweight"
It focuses on the recognition of handwritten documents on full-page/paragraph level.
We obtained the following results at paragraph level:

|  Dataset  |  cer |  wer  |
|:------------:|:----:|:-----:|
|      IAM     | 4.55 | 15.32 |
|   READ2016   | 3.94 | 16.06 |

and the following results for Tibetan documents at the main-text area:

|  Dataset  |  cer |  ser  |
|:------------:|:----:|:-----:|
|  BJK-185     | 3.13 | 7.33 |
|  LJK-200     | 1.91 | 4.48 |

Pretrained model weights are available [here](https://zenodo.org/records/16956903).

Table of contents:
1. [Getting Started](#Getting-Started)
2. [Datasets](#Datasets)
3. [Training And Evaluation](#Training-and-evaluation)

## Getting Started
Implementation has been tested with Python 3.8.

Clone the repository:

```
git clone https://github.com/Molorov/ParagraphTextOCR.git
```

Install the dependencies:

```
conda env create -f environment.yml
```


## Datasets
This section is dedicated to the datasets used in the paper: download and formatting instructions are provided 
for experiment replication purposes.

### IAM

#### Details

IAM corresponds to english grayscale handwriting images (from the LOB corpus).
The formatting script is provided by Denis Coquenet at [here](https://github.com/FactoDeepLearning/VerticalAttentionOCR)
The different splits are as follow:

|           | train | validation |  test |
|:---------:|:-----:|:----------:|:-----:|
| paragraph |  747  |     116    |  336  |

#### Download


- Register at the [FKI's webpage](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
- Download the dataset [here](https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database) 
- Move the following files into the folder Datasets/raw/IAM/
    - formsA-D.tgz
    - formsE-H.tgz
    - formsI-Z.tgz
    - lines.tgz
    - ascii.tgz

### READ 2016

#### Details
READ 2016 corresponds to Early Modern German RGB handwriting images.
The formatting script is provided by Denis Coquenet at [here](https://github.com/FactoDeepLearning/VerticalAttentionOCR)
The different splits are as follow:

|           | train | validation |  test |
|:---------:|:-----:|:----------:|:-----:|
|    line   | 8,349 |  1,040    | 1,138|
| paragraph |  1584 |     179    | 197 |

#### Download

- From root folder:

```
cd Datasets/raw
mkdir READ_2016
cd READ_2016
wget https://zenodo.org/record/1164045/files/{Test-ICFHR-2016.tgz,Train-And-Val-ICFHR-2016.tgz}
```


### Format the datasets

- Comment/Uncomment the following lines from the main function of the script "format_datasets.py" according to your needs and run it

```
if __name__ == "__main__":

    # format_IAM_line()
    # format_IAM_paragraph()

    # format_RIMES_line()
    # format_RIMES_paragraph()

    # format_READ2016_line()
    # format_READ2016_paragraph()
```

- This will generate well-formated datasets, usable by the training scripts.

## BJK-185 & LJK-200
BJK-185 corresponds to the Beijing edition of Kangyur, it is a dataset consists of woodblock printed Tibetan historical documents.
The formatted dataset is available at [here]().
LJK-200 corrresponds to the Lijiang edition of Kangyur.
The formatted dataset is available at [here](https://pan.baidu.com/s/1nG6u3yTrJADcwWvtnWp6Jw?pwd=965p) with access code: 965p


## Training And Evaluation
You need to have a properly formatted dataset to train a model, please refer to the section [Datasets](#Datasets). 

Two scripts are provided to train respectively line and paragraph level models: OCR/line_OCR/ctc/main_line_ctc.py and OCR/document_OCR/v_attention/main_pg_va.py

Training a model leads to the generation of output files ; they are located in the output folder OCR/line_OCR/ctc/outputs/#TrainingName or OCR/document_OCR/v_attention/outputs/#TrainingName.

The outputs files are split into two subfolders: "checkpoints" and "results". "checkpoints" contains model weights for the last trained epoch and for the epoch giving the best valid CER.
"results" contains tensorboard log for loss and metrics as well as text file for used hyperparameters and results of evaluation.

Training can use apex package for mix-precision and Distributed Data Parallel for usage on multiple GPU.

All hyperparameters are specified and editable in the training scripts (meaning are in comments).

Evaluation is performed just after training ending (training is stopped when the maximum elapsed time is reached or after a maximum number of epoch as specified in the training script)

## Citation

```bibtex
@ARTICLE{Coquenet2022,
    author={Coquenet, Denis and Chatelain, Clement and Paquet, Thierry},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
    title={End-to-end Handwritten Paragraph Text Recognition Using a Vertical Attention Network},
    year={2022},
    doi={10.1109/TPAMI.2022.3144899}
}
```

## License

This whole project is under Cecill-C license EXCEPT FOR the file "basic/transforms.py" which is under MIT license.
