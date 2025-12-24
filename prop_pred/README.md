# Training AttentiveFP property predictor models

## Setting up Environment
After setting up [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) execute the following steps: 
1. Create new conda environment and activate it:
```bash
conda create -n surfpro python=3.11

conda activate surfpro
```

2. Install required libraries:
```bash
pip install -r requirements.txt
```

## Training Property Predictor Models
1. Change the `train_configs.yaml` file with necessary changes. \
To train the single-property predictor model used in our work, set `props` argument to `['pCMC']` and `scaling` argument to `False`. \
To train multi-property predictor model used in our work set `props` to `['pCMC', 'AW_ST_CMC', 'Area_min']` and set `scaling` to `True`. \
**Note**: Change the `save_path` argument to different locations for multi and single models, unless you want the code to overwrite over the current files in your `save_path` folder.

2. Execut the following line: 
```bash
python3 main.py --config_file train_configs.yaml
``` 

The models are saved inside `{saved_path}/lightning_logs/` where `saved_path` is the path to folder set in `train_configs.yaml` file. 




## Acknowledgements
This project is based in part on components from the original
[SurfPro](https://github.com/BigChemistry-RobotLab/SurfPro) repository.
Only the parts relevant to our requirements were used. 