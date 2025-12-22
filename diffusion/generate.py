import torch
import torch_molecule
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import gc
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/model_pCMC_AW_ST_CMC_Area_min.pth')
    parser.add_argument('--n_gen', type=int, default=10)
    parser.add_argument('--props',nargs='+',type=str,default=['pCMC', 'AW_ST_CMC', 'Area_min'])
    parser.add_argument('--save_folder', type=str, default='../generated_data')
    parser.add_argument('--df_path', type=str, default='../data/surfpro_imputed_non_ionic.csv')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_folder, exist_ok=True)
    model = torch.load(args.model_path, weights_only=False)
    df = pd.read_csv(args.df_path)
    test_df = df[df['fold']==-1]
    y_val = test_df[args.props].to_numpy()
    smiles = test_df['SMILES'].to_numpy()


    data = defaultdict(list)
    mol_idx = 0
    smi_idx = 0
    for smi,y in tqdm(zip(smiles, y_val), total=len(smiles)):
        # print(mol_idx+1)
        mol_idx+=1
        data[f'smile_act'].append(smi)
        labels = np.tile(y,(args.n_gen,1))
        
        smi_idx+=1
        for idx,prop in enumerate(args.props):
            # print(prop)
            data[prop].append(y[idx])

        smi_gen = model.generate(labels=labels)

        
        for i,gen in enumerate(smi_gen):
            data[f'smile_gen_{i+1}'].append(gen)
            
            # print(smi_gen)

        data_df = pd.DataFrame(data)
        if len(args.props)==1:
            data_df.to_csv(f'{args.save_folder}/generated_single_10_mol_per_prop.csv', index=False)
        else:
            data_df.to_csv(f'{args.save_folder}/generated_multi_10_mol_per_prop.csv', index=False)

        torch.cuda.empty_cache()

        if mol_idx%10==0:
            del model
            gc.collect()
            torch.cuda.empty_cache()
            model = torch.load(args.model_path, weights_only=False)





