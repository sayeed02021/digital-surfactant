# Transformer Training & Generation Pipeline

This repository provides scripts to **fine-tune a Transformer model** and **generate predictions** in a conda-based environment.

---


## 1. Environment Setup

Make sure you have Conda / [Miniconda]((https://www.anaconda.com/docs/getting-started/miniconda/install)) installed.

Create the environment from the provided YAML file and activate it:

```bash
conda create -n trfm_opt python=3.7

conda activate trfm_opt
```

Install required libraries:
```bash
pip install -r requirements.txt
```

```bash
cd transformer
```

---
## 2. Model Training (Pre-training)

**NOTE:** Before running the code, download the CHEMBL matched molecular pairs data from here: [Link](https://drive.google.com/file/d/1_RdbfVLE5_x0yp1Wr7TIUnafaN2eorh6/view?usp=sharing). Move downloaded `.csv` file into `data_chembl/chembl_02` folder.

Execute the following lines to build the vocabulary and split data into train test split

For unconditional pre training go to configuration/config_default.py and set `properties = []`

```bash
python preprocess.py \
  --input-data-path data_chembl/chembl_02/mmp_prop.csv
```

Use the following command to fine-tune a Transformer model using a pretrained checkpoint:

```bash
python train.py \
  --pretrain \
  --data-path data_chembl/chembl_02/ \
  --save-directory pretrain_transformer \
  --batch-size 128 \
  --num-epoch 200 
```

-----

## 3. Fine-tuning

### Single Property

Execute the following lines to fine-tune a Transformer model using a pretrained checkpoint:

1. For single property go to `configuration/config_default.py` and select the `properties = ['pCMC']`

2. Execute the following lines
```bash
python preprocess.py \
  --input-data-path data_single/final_mmps_single.csv 
```
```bash
python train.py \
  --finetune \
  --pretrained-model-path pretrain_transformer/checkpoint/model_23.pt \
  --data-path data_single \
  --save-directory fine_tune_transformer_single\
  --batch-size 128 \
  --num-epoch 300
```

**Key Arguments**

* `--finetune` : Enables fine-tuning mode
* `--pretrained-model-path` : Path to the pretrained checkpoint
* `--data-path` : Directory containing training/validation data
* `--save-directory` : Directory where checkpoints and logs are saved
* `--batch-size` : Training batch size
* `--num-epoch` : Number of training epochs


Checkpoints are saved under: `experiments/fine_tune_transformer_single/checkpoint/`

---

### Multi Property

Use the following command to fine-tune a Transformer model using a pretrained checkpoint:

1. For multi property go to configuration/config_default.py and select the `properties = ['pCMC','Area_min','ST_AW_CMC']`
2. Execute following lines:
```bash
python preprocess.py \
  --input-data-path data_multi/final_mmps_multi.csv 
```
```bash
python train.py \
  --finetune \
  --pretrained-model-path pretrain_transformer/checkpoint/model_23.pt \
  --data-path data_multi \
  --save-directory fine_tune_transformer_multi\
  --batch-size 128 \
  --num-epoch 300
  
```

**Key Arguments**

* `--finetune` : Enables fine-tuning mode
* `--pretrained-model-path` : Path to the pretrained checkpoint
* `--data-path` : Directory containing training/validation data
* `--save-directory` : Directory where checkpoints and logs are saved ; check `eperiments/fine_tune_transformer_multi`
* `--batch-size` : Training batch size
* `--num-epoch` : Number of training epochs


Checkpoints are saved under: `experiments/fine_tune_transformer_multi/checkpoint/`

---

## 4. Generation / Evaluation

After training, generate predictions on the test sets

1. For single property go to `configuration/config_default.py` and select `properties = ['pCMC']` or for multi property select `properties = ['pCMC','Area_min','ST_AW_CMC']`
2. Below is an example of generating using multi property model:
```bash
python generate.py \
  --data-path data_multi/ \
  --test-file-name test \
  --model-path experiments/fine_tune_transformer_multi/checkpoint \
  --save-directory evaluation_transformer_multi \
  --epoch 100
```

**Key Arguments**
* `--data-path` : folder where test file is located
* `--test-file-name test` : Name of the test split file
* `--model-path` : Path to trained model checkpoints
* `--save-directory` : Directory where generated outputs are stored
* `--epoch 100` : Epoch number of the checkpoint to load
