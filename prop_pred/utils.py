import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import defaultdict
import yaml
import glob
import os
from model import LitAttentiveFP
import argparse
import pytorch_lightning as pl

def ensemble_results(df, props, args):
    # df = pd.read_csv(f'{args.save_path}/all_preds.csv')
    df = df.groupby("smiles").agg(
        **{f'{prop}_target': (f'{prop}_target', 'first') for prop in props},
        **{f'{prop}_pred': (f'{prop}_pred', 'mean') for prop in props}
    ).reset_index()

    df.to_csv(f'{args.save_path}/all_preds.csv', index=False)
    print(df.shape)
    print('\n')
    data = defaultdict(list)
    for prop in props:
        pred = torch.tensor(df[f'{prop}_pred'].to_numpy())
        target = torch.tensor(df[f'{prop}_target'].to_numpy())
        
        # Filter out nan values for this property
        mask = ~torch.isnan(target)
        pred_clean = pred[mask]
        target_clean = target[mask]
        
        # Calculate MAE and RMSE on the clean data
        mae = F.l1_loss(pred_clean, target_clean).item()
        rmse = torch.sqrt(F.mse_loss(pred_clean, target_clean)).item()

        print(f'{prop}: MAE: {mae} | RMSE: {rmse}\n')

        data['prop'].append(prop)
        data['MAE'].append(mae)
        data['RMSE'].append(rmse)
    df = pd.DataFrame(data)
    df.to_csv(f'{args.save_path}/final_results.csv', index = False)

def get_preds(loader, args):
    # print("Length Loader: ", len(loader))
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=None,
        enable_progress_bar=False
    )
    all_data = []
    model_path = f'{args.model_path}/lightning_logs/version'
    for fold in range(10):
        if os.path.exists(f'{model_path}_{fold}'): # check if this fold exists or not
            model_path_final = glob.glob(f'{model_path}_{fold}/checkpoints/*.ckpt')[-1]
            config_path = f'{model_path}_{fold}/config.yaml'
        else:
            break
        with open(config_path, 'r') as f:
            config=yaml.safe_load(f)
        config_args = argparse.Namespace(**config)
        model = LitAttentiveFP.load_from_checkpoint(
            model_path_final, args=config_args
        )

        results=trainer.predict(model, loader) # list of length = batch_size
        ## flatten the list
        out = {}
        for k in results[0].keys():
            values = [d[k] for d in results]

            if isinstance(values[0], torch.Tensor):
                out[k] = torch.cat(values, dim=0)
            elif isinstance(values[0], np.ndarray):
                out[k] = np.concatenate(values, axis=0)
            elif isinstance(values[0], list):
                out[k] = sum(values, [])
            else:
                out[k] = values
        # print("Output length: ", len(out['preds']))   
        all_data.append(out) # list of dictionaries containing predictions of different models
    # print(len(all_data[0]['preds']))
    final_data = defaultdict(list)

    for result in all_data:
        for key in result.keys():
            final_data[key].append(result[key])
    
    # print(final_data['index'])

    return final_data


