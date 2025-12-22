# Conditional Generation using GraphDiT

## Setting up Environment
After setting up [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) execute the following steps: 
1. Create new conda environment and activate it:
```bash
conda create -n torch_mol python=3.11

conda activate torch_mol
```

2. Install required libraries:
```bash
pip install -r requirements.txt
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

## Generating from trained models

After the models are saved in `models/` to generate new molecules from the test set property values execute below lines: 
1. To generate from Diff-Single model:
```python
python3 generate.py \
    --model_path models/model_pCMC.pth \
    --n_gen 10 \
    --props pCMC \
    --save_folder ../generated_data \
    --df_path ../data/surfpro_imputed_non_ionic.csv
```
2. To generate molecules using Diff-Multi model: 
```python
python3 generate.py \
    --model_path models/model_pCMC_AW_ST_CMC_Area_min.pth \
    --n_gen 10 \
    --props pCMC AW_ST_CMC Area_min \
    --save_folder ../generated_data \
    --df_path ../data/surfpro_imputed_non_ionic.csv
```