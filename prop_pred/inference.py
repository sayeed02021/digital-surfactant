from torch_geometric.loader import DataLoader
from sklearn.preprocessing import RobustScaler

import pandas as pd
import numpy as np
import argparse

from dataset import SMILESDataset
from utils import get_preds

import os
import logging
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--props',nargs='+',type=str,default=['pCMC', 'AW_ST_CMC', 'Area_min'], help='Properties to be evaluated')
    parser.add_argument('--model_path', type=str, default='multi', help='Path to saved models')
    parser.add_argument('--save_path', type=str, default='../generated_data/diff_multi', help='Path to stored predictions')
    parser.add_argument('--csv_path', type=str, default='../generated_data/diff_multi/generated_all.csv', help='Path to saved .csv file')
    parser.add_argument('--method', type=str, choices=['trfm', 'diff'],required=True, help='Check if the data has been generated from transformer or diffusion models')
    parser.add_argument('--train_data_path', type=str, default='data/surfpro_train.csv', help='Path to original training data(used for fitting RobustScaler())')
    args = parser.parse_args()
    return args



def infer_one_file_trfm(df, fold_idx, args, scaler=None):
    df = df.dropna(subset=[f"Predicted_smi_{fold_idx+1}"]).reset_index(drop=True)
    gen_smiles = df[f'Predicted_smi_{fold_idx+1}'].to_numpy()
    target_prop_cols = [f'Target_Mol_{p}' for p in args.props]
    source_prop_cols = [f'Source_Mol_{p}' for p in args.props]
    target_smiles = df['Target_Mol'].to_numpy()
    source_smiles = df['Source_Mol'].to_numpy()
    target_props = df[target_prop_cols].to_numpy()
    source_props = df[source_prop_cols].to_numpy()
    source_data = {
        'source_smile': source_smiles,
        'source_prop': source_props,
        'target_smile': target_smiles,
        'target_prop': target_props
    }
    if scaler is not None:
        scale_data = True
    else:
        scale_data = False


    dataset = SMILESDataset(
        smiles = gen_smiles,
        index = np.arange(len(gen_smiles)),
        mode = 'test',
        feat = args.props,
        y_val = None,
        fold_values=None,
        scaler=scaler,
        scale_data=scale_data,
        source_data=source_data
    )
    # print("Data Length: ", len(dataset))
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    pred_dict = get_preds(loader, args) # dictonary having different model outputs in list format
    preds = pred_dict['preds']
    preds_unscaled = []
    for p in preds:
        # p = p.numpy()
        if scaler is not None:
            p_unscaled = scaler.inverse_transform(p)
            preds_unscaled.append(p_unscaled)
        else:
            preds_unscaled.append(p)
    pred_dict['preds'] = preds_unscaled

    final_preds = {}
    preds = pred_dict['preds']
    target_prop = np.array(pred_dict['target_prop'][0])
    # print(target_prop.shape)
    source_prop = np.array(pred_dict['source_prop'][0])
    preds = np.array(preds)
    # print(preds.shape[0])
    ensemble_preds = preds.mean(axis=0)
    for idx, prop in enumerate(args.props):
        final_preds[f'pred_{prop}'] = ensemble_preds[:,idx].tolist()
        final_preds[f'target_{prop}'] = target_prop[:,idx].tolist()
        final_preds[f'source_{prop}'] = source_prop[:,idx].tolist()
    final_preds['gen_smiles'] = pred_dict['smiles'][0]
    final_preds['target_smiles'] = pred_dict['target_smile'][0]
    final_preds['source_smiles'] = pred_dict['source_smile'][0]
    final_df = pd.DataFrame(final_preds)

    cols_order = ['source_smiles']+[f'source_{p}' for p in args.props]+['target_smiles']+[f'target_{p}' for p in args.props] + ['gen_smiles'] + [f'pred_{p}' for p in args.props]
    final_df = final_df[cols_order]
    final_df.to_csv(f'{args.save_path}/predicted_smiles_fold_{fold_idx+1}.csv', index=False)


    
def infer_one_file_diff(df, fold_idx, args, scaler=None):
    df = df.dropna(subset=[f"smile_gen_{fold_idx+1}"]).reset_index(drop=True)
    gen_smiles = df[f'smile_gen_{fold_idx+1}'].to_numpy()
    target_smiles = df[f'smile_act'].to_numpy()
    target_props = df[args.props].to_numpy()

    source_data = {
        'target_smile': target_smiles,
        'target_prop': target_props
    }
    if scaler is not None:
        scale_data = True
    else:
        scale_data = False
    
    dataset = SMILESDataset(
        smiles = gen_smiles,
        index = np.arange(len(gen_smiles)),
        mode = 'test',
        feat = args.props,
        y_val = None,
        fold_values=None,
        scaler=scaler,
        scale_data=scale_data,
        source_data=source_data
    )

    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    
    pred_dict = get_preds(loader, args)


    preds = pred_dict['preds']
    preds_unscaled = []
    for p in preds:
        # p = p.numpy()
        if scaler is not None:
            p_unscaled = scaler.inverse_transform(p)
            preds_unscaled.append(p_unscaled)
        else:
            preds_unscaled.append(p)
    pred_dict['preds'] = preds_unscaled

    
    final_preds = {}
    preds = pred_dict['preds']
    target_prop = np.array(pred_dict['target_prop'][0])

    preds = np.array(preds)

    ensemble_preds = preds.mean(axis=0)
    for idx, prop in enumerate(args.props):
        final_preds[f'pred_{prop}'] = ensemble_preds[:,idx].tolist()
        final_preds[f'target_{prop}'] = target_prop[:,idx].tolist()
    final_preds['gen_smiles'] = pred_dict['smiles'][0]
    final_preds['target_smiles'] = pred_dict['target_smile'][0]
    final_df = pd.DataFrame(final_preds)

    cols_order = ['target_smiles']+[f'target_{p}' for p in args.props] + ['gen_smiles'] + [f'pred_{p}' for p in args.props]
    final_df = final_df[cols_order]
    final_df.to_csv(f'{args.save_path}/predicted_smiles_fold_{fold_idx+1}.csv', index=False)


def main(args):
    df = pd.read_csv(args.csv_path)
    columns = df.columns
    if args.method=='diff':
        generated_cols = [c for c in columns if c.startswith('smile_gen')]
        target_prop_cols = args.props
        target_smi_col = 'smile_act'
        source_prop_cols = None
        source_smi_col = None
        if len(generated_cols)==0:
            raise ValueError('Incorrect method argument provided OR Cannot read generated file properly')
    else:
        generated_cols = [c for c in columns if c.startswith('Predicted_smi')]
        target_prop_cols = [f'Target_Mol_{p}' for p in args.props]
        target_smi_col = 'Target_Mol'
        source_prop_cols = [f'Source_Mol_{p}' for p in args.props]
        source_smi_col = 'Source_Mol'

        if len(generated_cols)==0:
            raise ValueError('Incorrect method argument provided OR Cannot read generated file properly')
    
    if len(args.props)>1:
        # use scaler 
        train_df = pd.read_csv(args.train_data_path)
        y_val = train_df[args.props].to_numpy()
        assert y_val.shape[1]==len(args.props)
        scaler = RobustScaler()
        scaler.fit(y_val)
    else:
        scaler=None


    for idx,gen_col in enumerate(generated_cols):
        if source_prop_cols is None:
            filtered_columns = [target_smi_col]+ target_prop_cols+ [gen_col]
        else:
            filtered_columns = [source_smi_col]+ source_prop_cols + [target_smi_col] + target_prop_cols + [gen_col]
        
        filtered_df = df[filtered_columns]
        
        if args.method=='trfm':
            infer_one_file_trfm(
                df = filtered_df,
                fold_idx = idx,
                args=args,
                scaler = scaler
            )
        else:
            infer_one_file_diff(
                df = filtered_df, 
                fold_idx = idx,
                args=args,
                scaler = scaler
            )


if __name__=='__main__':
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    main(args)


