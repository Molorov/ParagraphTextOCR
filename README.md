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

### IAM & READ2016

#### Details

IAM corresponds to english grayscale handwriting images (from the LOB corpus).

READ 2016 corresponds to Early Modern German RGB handwriting images.

The formatting script is provided by Denis Coquenet at [here](https://github.com/FactoDeepLearning/VerticalAttentionOCR)

The different splits are as follow:

|           | train | validation |  test |
|:---------:|:-----:|:----------:|:-----:|
|   IAM     |  747  |     116    |  336  |
| READ2016  |  1584 |     179    | 197   |

#### IAM Download


- Register at the [FKI's webpage](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
- Download the dataset [here](https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database) 
- Move the following files into the folder Datasets/raw/IAM/
    - formsA-D.tgz
    - formsE-H.tgz
    - formsI-Z.tgz
    - lines.tgz
    - ascii.tgz


#### READ2016 Download

- From root folder:

```
cd Datasets/raw
mkdir READ_2016
cd READ_2016
wget https://zenodo.org/record/1164045/files/{Test-ICFHR-2016.tgz,Train-And-Val-ICFHR-2016.tgz}
```


#### Format the datasets

- Comment/Uncomment the following lines from the main function of the script "format_datasets.py" according to your needs and run it

```
if __name__ == "__main__":

    # format_IAM_line()
    # format_IAM_paragraph()

    # format_READ2016_line()
    # format_READ2016_paragraph()
```

- This will generate well-formated datasets, usable by the training scripts.

### BJK-185 & LJK-200

BJK-185 corresponds to the Beijing edition of Kangyur, while the LJK-200 corrresponds to the Lijiang edition of Kangyur. They both are
datasets consists of woodblock printed Tibetan historical documents.

The formatted BJK-185 is available at [here](https://pan.baidu.com/s/1X_scsIvnpzV00_DRnYFGcg?pwd=awjr) with access code: awjr

The formatted LJK-200 is available at [here](https://pan.baidu.com/s/1nG6u3yTrJADcwWvtnWp6Jw?pwd=965p) with access code: 965p


## Training And Evaluation
You need to have a properly formatted dataset to train a model, please refer to the section [Datasets](#Datasets). 

The following script is provided to train and evaluate the model: 
OCR/document_OCR/comp_ctc/main_pg_cc.py

The configuration files are stored in the folder: 
config_cc/#ConfigName.yaml

Training a model leads to the generation of output files ; they are located in the output folder:
OCR/document_OCR/comp_ctc/outputs/#ConfigName.

The outputs files are split into two subfolders: "checkpoints" and "results". "checkpoints" contains model weights for the last trained epoch and for the epoch giving the best valid CER.
"results" contains tensorboard log for loss and metrics as well as text file for used hyperparameters and results of evaluation.

Training can use apex package for mix-precision and Distributed Data Parallel for usage on multiple GPU.

- The following command is used to train the model with corresponding configuration:
  
  python main_pg_cc.py --config ../../../config_cc/#ConfigName.yaml

- The following command is used to evaluate the model after the training is complete:
  
  python main_pg_cc.py --config ../../../config_cc/#ConfigName.yaml --test --line_match
  where the "line_match" argument is optional.

Note that when training on the Tibetan datasets, it is necessary to first pre-train the model without syllable points, since these are often poorly labeled and tend to hinder convergence.

- For exmaple, one need to first run the pretraining configuration:
  
  python main_pg_cc.py --config ../../../config_cc/BJK_185+nosp.yaml
  
- After the pretraining is complete, then run the configuration:

  python main_pg_cc.py --config ../../../config_cc/BJK_185.yaml

    
## Citation

```bibtex
@ARTICLE{Mao2025,
    author={Mao, Leer and Wang, Weilan and Li, Qiaoqiao},
    journal={Pattern Recognition},
    title={Composite-CTC for Full-page Handwritten Text Recognition: Non-autoregressive, Minimal Supervision, and Lightweight},
    year={2025},
    doi={}
}
```

## License

This whole project is under Cecill-C license EXCEPT FOR the file "basic/transforms.py" which is under MIT license.
