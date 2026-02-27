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
- **Paper**: TBA
- **Guideline**: [Link](https://drive.google.com/drive/folders/1-atFWh69MJ7vsAsa_1WQm6YhYe-4IGa3?usp=sharing)
- **Dataset**: [hf](https://huggingface.co/datasets/weerayut/thai-english-financial-ner)
- **Colab**: [Link](https://colab.research.google.com/drive/1v_cs14tJp9yY4HwWJ2C3IjFLSb77PC9F?usp=share_link)
- **Checkpoints**: [Link](https://drive.google.com/open?id=1-AM4QF9R4q5di9JZu_qRJBXz_XTL27Wd&usp=drive_fs)


## Experimental setup ###
| Pre-trained model |
| --- |
| WangchanBART-BASE |
| WangchanBART-LARGE |
| WangchanBART-Large-Finance |
| XLM-RoBERTa-base |
| XLM-RoBERTa-large |


## Dataset statistics ###
Language | Documents | Mentions | Tokens |
| --- | --- | --- | --- |
| ALL | 48,376 | 252,904 | 3,871,094
| TH | 3,280 | 120,274 |1,364,839 |
| EN | 11,428 | 132,630 | 2,506,255 |

| Source | Language | Documents | Mentions | Tokens |
| --- | --- | --- | --- | --- |
| Pachachat-finance | TH | 740 | 32,390 |401,737 |
| PostToday-finance | TH | 1,000 | 34,408 | 477,372 |
| Kaohoon | TH | 1,140 | 40,301 | 370,280 |
| Kasikorn | TH | 400 | 13,175 | 146,907 |
| Reddit-investing | EN | 5,567 | 71,589 | 1,410,788 |
| Reddit-robinhood | EN | 322 | 2,179 | 42,976 |
| Reddit-stocks | EN | 5,539 | 58,862 | 1,052,491 |

## Kappa agreement score ###
| Items | Score |
| --- | --- |
| Full | 0.78 |
| Thai | 0.81 |
| English | 0.77 |


## Table experimental results ###
| Model | Precision | Recall | F1(%) |
| --- | :---: | :---: | :---: | 
| **Full dataset** |
| WangchanBART-BASE  | 80.40 | 87.02 | 83.58 |
| WangchanBART-LARGE  | 81.74 | 88.20 | 84.84 |
| WangchanBART-Large-Finance  | 78.97 | 85.82 | 82.25 |
| XLM-RoBERTa-base  | 82.44 | 86.72 | 84.53 |
| XLM-RoBERTa-large  |84.00 | 87.70 | 85.81 |

## Hyperparameters ###
| Hyperparameter | Value |
| --- | --- |
| Learning rate | 1e-4, 1e-5, 1e-6, 1e-7|
| Dropout | 0.5 |
| Seed | 42 |
| Batch size | 8 |

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
