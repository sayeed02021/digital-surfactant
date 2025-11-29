"""
Code for inference of generated molecules using trained saved models

"""
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from dataset import SMILESDataset
from model import LitAttentiveFP

from sklearn.preprocessing import RobustScaler

import pandas as pd
import numpy as np
import os
import glob
import yaml
import argparse
from pytorch_lightning import seed_everything
from collections import defaultdict

from utils import ensemble_results


import logging
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)



props = ['pCMC', 'AW_ST_CMC', 'Area_min']
# props = ['pCMC']
# props = ['pCMC', 'pCMC', 'pCMC']

save_path = 'diff_gen_mols_5'

if len(props)>1:
    scaling=True
    csv_paths = glob.glob(f'{save_path}/data_multi/generated_*.csv')
    model_root_path = 'scaled/new'
    save_path = f'{save_path}/data_multi'
    os.makedirs(save_path, exist_ok=True)

else:
    model_root_path = 'unscaled/pCMC'
    scaling=False
    csv_paths = glob.glob(f'{save_path}/data_single/generated_*.csv')
    save_path = f'{save_path}/data_single'


with open(f'{model_root_path}/config.yaml', "r") as f:
    config = yaml.safe_load(f)
args = argparse.Namespace(**config)

if scaling:
    train_df = pd.read_csv('../data/surfpro_train.csv')
    train_props = train_df[props].to_numpy()
    scaler = RobustScaler()
    scaler = scaler.fit(train_props)
else:
    scaler=None


def inference_for_one_file(file_path, idx):
    df = pd.read_csv(file_path)
    initial_shape = df.shape[0]
    df['index'] = np.arange(len(df)).tolist()
    df = df.dropna(subset=["smile_gen"]).reset_index(drop=True)
    generated_smiles = df[f'smile_gen'].to_numpy()
    indices = df['index'].to_numpy()
    not_gen = initial_shape-len(generated_smiles)
    print(f"Pred: {idx}| {not_gen}/{initial_shape} Molecules not Generated")
    
    target_smiles = df[f'smile_act'].to_numpy()
    # source_smiles = df[f'smile_source'].to_numpy()
    # source_props = [f'source_{prop}' for prop in props]
    # source_values = df[source_props].to_numpy().tolist()
    # if len(props)==1:
    #     source_values = source_values.unsqueeze(-1)
    
    target_props = [f'target_{p}' for p in props]
    columns = df.columns.tolist()
    for tp in target_props:
        if tp not in columns:
            df[tp] = -1*np.ones(len(df))

    target_values = df[target_props].to_numpy()
    print(target_values.shape)
    dataset = SMILESDataset(
        smiles = generated_smiles,
        index=indices,
        feat = props,
        y_val = target_values,
        scale_data=scaling,
        scaler=scaler,
        root=f'{save_path}/pth_data',
        source_data={
            'target_smiles': target_smiles,
            # 'source_smiles': source_smiles,
            # 'source_y': source_values
        },
        mode='test',

    )

    

    

    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=None,
        enable_progress_bar=False,
        
    )

    
    all_data = defaultdict(list)

    for fold in range(10):
        model_path = glob.glob(f'{model_root_path}/lightning_logs/version_{fold}/checkpoints'+'/*.ckpt')[-1]
        if len==1:
            model = LitAttentiveFP.load_from_checkpoint(model_path)
        else:
            model = LitAttentiveFP.load_from_checkpoint(model_path,args=args)
        results = trainer.predict(model, loader)
        for result in results:
            for key in result.keys():
                if key=='preds' or key=='targets' or key=='source_y':
                    continue
                if isinstance(result[key], list):
                    all_data[key].extend(result[key])
                else:
                    all_data[key].extend(result[key].tolist())
            
            targets = result['targets']
            preds = result['preds']
            if scaling:
                targets = scaler.inverse_transform(targets)
                preds = scaler.inverse_transform(preds)
            
            # source_y = np.array(result['source_y'])
            for i,p in enumerate(props):

                target_val = targets[:,i]
                pred_val = preds[:,i]
                
                # source_val = source_y[:,i]
                if f'target_{p}' in columns:
                    all_data[f'target_{p}'].extend(target_val.tolist())
                all_data[f'pred_{p}'].extend(pred_val.tolist())
                # all_data[f'source_{p}'].extend(source_val.tolist())

    all_df = pd.DataFrame(all_data)
    # all_df.to_csv('trfm_gen_mols/check_data.csv', index=False)
    ensemble_results = all_df.groupby("index").agg(
        # source_smiles=('source_smiles', 'first'),
        # **{f'source_{prop}': (f'source_{prop}', 'first') for prop in props},
        target_smiles=('target_smiles', 'first'),
        **{f'target_{prop}': (f'target_{prop}', 'first') for prop in props if f'target_{prop}' in columns},
        gen_smiles=('smiles', 'first'),
        **{f'pred_{prop}': (f'pred_{prop}', 'mean') for prop in props},
    ).reset_index()

    ensemble_results.to_csv(f'{save_path}/pred_molecules_{idx}.csv', index=False)

        
        # for key,val in all_data.items():
        #     print(len(val))






if __name__=='__main__':
    for i in range(10):
        # if i==4:
        #     continue
        file_path = csv_paths[i]
        inference_for_one_file(file_path, i+1)
        