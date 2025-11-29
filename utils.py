import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import defaultdict

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




