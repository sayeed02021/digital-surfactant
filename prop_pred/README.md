# Training AttentiveFP property predictor models

## Setting up Environment
After setting up [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) execute the following steps: 
1. Create new conda environment and activate it:
```bash
conda create -n surfpro python=3.10

conda activate surfpro
```

2. Install required libraries:
```bash
pip install -r requirements.txt
```

```bash
cd prop_pred
```
---
## Training Property Predictor Models
1. Change the `train_configs.yaml` file with necessary changes. \
To train the single-property predictor model used in our work, set `props` argument to `['pCMC']` and `scaling` argument to `False`. \
To train multi-property predictor model used in our work set `props` to `['pCMC', 'AW_ST_CMC', 'Area_min']` and set `scaling` to `True`. \
**Note**: Change the `save_path` argument to different locations for multi and single models, unless you want the code to overwrite over the current files in your `save_path` folder.

2. Execut the following line: 
```bash
python3 main.py --config_file train_configs.yaml
``` 
**Key Arguments in train_configs.yaml**
* `num_epochs`: Maximum number of epochs to train a model per fold
* `batch_size`: Training and testing batch size
* `n_folds`: Number of folds in the dataset
* `patience`: Number of epochs to wait before implementing early stopping
* `lr`: Learning rate to use if learning rate tuning is not used
* `stop_early`: If set to False, then trains all the models for entire 500 epochs
* `scaling`: If set to True, then uses RobustScaler() to scale all the property values(useful in multi-property case)
* `tuning`: Tunes learning rate per fold if set to True
* `save_path`: Path to save the models and final predictions of test set
* `SEED`: Sets the seed of each run
* `props`: The properties from the .csv file on which to train. If training single property model, include the property inside a list. Example: `props: ['pCMC']`

The rest of the parameters are hyperparameters of the AttentiveFP network. Check out the [AttentiveFP documentation](https://pytorch-geometric.readthedocs.io/en/2.5.2/generated/torch_geometric.nn.models.AttentiveFP.html) from torch_geometric for more details. 

The models are saved inside `{saved_path}/lightning_logs/`.

---
## Performing Inference on Generated SMILES

The trained property predictor models are used to infer the properties of molecules generated using diffusion or transformer.

Both diffusion and transformer generate all molecules into a single `.csv` file that can be found inside the `transformer/experiments` folder(in case of transformers) or the `generated_data/` folder (in case of diffusion).

To get the predicted properties of generated molecules make use of the `inference.py` script in present inside the `prop_pred/` folder. The instructions for getting the property predictions for `Diff-Multi` case is shown below: 

```bash
python3 inference.py \
--props pCMC AW_ST_CMC Area_min \
--model_path multi \
--save_path ../generated_data/diff_multi \
--csv_path ../generated_data/diff_multi/generated_all.csv \
--method diff \
--train_data_path ../data/surfpro_train.csv
```

**Key Arguments**
* `--props`: Enter the property values that you are predicting. For single-case enter `pCMC` while for multi-case enter `pCMC AW_ST_CMC Area_min`
* `--model_path`: The path to folder that contains the `lightning_logs/` folder. The property predictor models are inside `--model_path/lightning_logs/`. **Don't** include `lightning_logs` into this path.
* `--save_path`: Path to folder where you will be storing your predicted output `.csv` files.
* `--method`: Should be either `trfm` or `diff`. If the molecules have been generated using diffusion models choose `diff`, if generated using transformers choose `trfm`.
* `--trained_data_path`: Path to the training dataset that the property predictor models were trained on. Needed for fitting the property value scaler for multi-property case



# Computing metrics
To compute the metrics for generated models, first install [Molecular Sets](https://github.com/molecularsets/moses) library inside your current environment, either through pip or manually as show in their README file.

Run the `metrics.py` file. An example use is shown below: 
```bash
python3 metrics.py \
--method trfm
--pred_folder ../generated_data_trfm_single \ 
--props pCMC \
--train_path ../data/surfpro_train.csv \
--mol_type non-ionic
```
**Key Arguments**
* `--props`: Enter the property values that you are predicting. For single-case enter `pCMC` while for multi-case enter `pCMC AW_ST_CMC Area_min`
* `--method`: Should be either `trfm` or `diff`. If the molecules have been generated using diffusion models choose `diff`, if generated using transformers choose `trfm`.
* `--train_path`: Path to the training dataset that the property generative models were trained on(hard coded for surfpro_train.csv). 
* `mol_type`: Type of molecules(ionic/non-ionic) molecules that are being generated
* `--pred_folder`: Path to folder that contains the .csv files with generated molecules and their predictions



## Acknowledgements
This project is based in part on components from the original
[SurfPro](https://github.com/BigChemistry-RobotLab/SurfPro) repository.
Only the parts relevant to our requirements were used. 