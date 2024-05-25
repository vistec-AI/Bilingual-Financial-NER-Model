# Bilingual-Financial-NER-Model

## Setup environment ##
```bash
    conda create -n cmdf python=3.9 numpy matplotlib
    conda activate cmdf
    pip install numpy
    pip install matplotlib
    pip install torch==2.0.1+cu117
    pip install transformers==4.33.3
    pip install tabulate
    pip install seqeval
    pip install -q gradio 
```

## Reference ###
- **Paper**: https://www.overleaf.com/
- **Code**: xxx
- **Dataset**: xxx
- **Guideline**: xxx
- **Checkpoints**: xxx


## Experimental setup ###
| Pre-trained model |
| --- |
| XLM-RoBERTa-base |
| XLM-RoBERTa-large |
| WangchanBART-BASE |
| WangchanBART-LARGE |
| WangchanBART-Large-Finance |


## Data collection ###
| Source | Language | Mentions | Tokens | Documents |
| --- | --- | --- | --- | --- |
| Total | ALL | 249,858| XXX | XXX |
| - | TH | XXX | XXX | XXX |
| - | EN | XXX | XXX | XXX |

| Source | Language | Mentions | Tokens | Documents |
| --- | --- | --- | --- | --- |
| Kaohoon | TH | XXX | XXX | XXX |
| Kasikorn-stock | TH | XXX | XXX | XXX |
| Kasikorn-trading | TH | XXX | XXX | XXX |
| Pachachat-finance | TH | XXX | XXX | XXX |
| PostToday-finance | TH | XXX | XXX | XXX |
| Reddit-investing | EN | XXX | XXX | XXX |
| Reddit-robinhood | EN | XXX | XXX | XXX |
| Reddit-stocks | EN | XXX | XXX | XXX |


## Kappa agreement score ###
| Items | Score |
| --- | --- |
| Full | 0.78 |
| Thai | 0.81 |
| English | 0.77 |


## Dataset statistics ###
- **Finance-NER-dataset (Standard split: 70/10/20)**:
    | Dataset | Mentions | Tokens | Documents |
    | --- | --- | --- | --- |
    | Train | XXX | XXX | XXX |
    | Dev | XXX | XXX | XXX |
    | Test | XXX | XXX | XXX |


## Table experimental results ###
| Model | F1 | Precision | Recall | lr | name |
| --- | :---: | :---: | :---: | :---: | :---: |
| **Full dataset** |
| XLM-RoBERTa-base  | XX.XX | XX.XX | XX.XX | XXX | XXX |
| XLM-RoBERTa-large  | XX.XX | XX.XX | XX.XX | XXX | XXX |
| WangchanBART-BASE  | XX.XX | XX.XX | XX.XX | XXX | XXX |
| WangchanBART-LARGE  | XX.XX | XX.XX | XX.XX | XXX | XXX |
| WangchanBART-Large-Finance  | XX.XX | XX.XX | XX.XX | XXX | XXX |

## Hyperparameters ###
| Hyperparameter | Value |
| --- | --- |
| Learning rate | 1e-4, 1e-5, 1e-6, 1e-7|
| Dropout | 0.5 |
| Seed | 42 |
| Batch size | 8 |
| Epochs | 100 |
| Early stopping | 8 |
| Eval per epoch | 4 |


## Train script
```bash
    python main.py \
        --lr 1e-5 \
        --epochs 100 \
        --mode train \
        --batch_size 8 \
        --name baseline \
        --early_stopping 8 \
        --checkpoint_dir ../checkpoints \
        --pretrained xlm-roberta-base \
        --data_path_train .../train.conll \
        --data_path_dev .../dev.conll \
        --data_path_test .../test.conll \
        --resume None
```


## Test script
Output will be saved in a checkpoint directory/{model_name}/
```bash 
    python main.py \
        --mode test \
        --pretrained xlm-roberta-large \
        --resume ...checkpoint/xlm-roberta-large/dir_name1 \
        --data_path_test data/toy.conll
```