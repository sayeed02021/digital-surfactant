# Conditional Generation using GraphDiT

## Setting up Environment
After setting up [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) execute the following steps: 
1. Create new conda environment and activate it:
```bash
conda create -n diff python=3.11

conda activate diff
```

2. Install required libraries:
```bash
pip install -r requirements.txt
```

```bash
cd diffusion
```

## Training the Diffusion Models

After install necessary libraries, execute following line to train the Diff-Single model: 
```python
python3 train.py \
    --props pCMC \
    --df_path ../data/surfpro_imputed_non_ionic.csv \
    --save_folder models \
    --batch_size 20 \
    --n_epoch 10000
```

and execute below line to train the Diff-Multi model:
```python
python3 train.py \
    --props pCMC AW_ST_CMC Area_min \
    --df_path ../data/surfpro_imputed_non_ionic.csv \
    --save_folder models \
    --batch_size 20 \
    --n_epoch 10000
```

**Key Arguments**
* `--props`: The properties of which model will be conditioned
* `df_path`: The path to the .csv dataset file
* `--save_folder`: The folder where you want to save your files
* `--batch_size`: Training batch size
* `--n_epoch`: Number of Training Epochs 

## Generating from trained models

After the models are saved in `models/` to generate new molecules from the test set property values execute below lines: 
1. To generate from Diff-Single model:
```python
python3 generate.py \
    --model_path models/model_single.pth \
    --n_gen 10 \
    --props pCMC \
    --save_folder ../generated_data \
    --df_path ../data/surfpro_imputed_non_ionic.csv
```
2. To generate molecules using Diff-Multi model: 
```python
python3 generate.py \
    --model_path models/model_multi.pth \
    --n_gen 10 \
    --props pCMC AW_ST_CMC Area_min \
    --save_folder ../generated_data \
    --df_path ../data/surfpro_imputed_non_ionic.csv
```

**Key Arguments**
* `--model_path`: Path to saved trained model.
* `--n_gen`: Number of molecules you want to generate from a single set of property values.
* `--props`: Property values the trained model has been conditioned upon.
* `--save_folder`: Path to folder where you want to save your generated molecules in. A single `.csv` stores the molecules generated from a model.
* `--df_path`: Path to dataset that contains the input property values using which you want to generate your molecules.
