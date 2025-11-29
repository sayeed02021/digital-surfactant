import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch_molecule
from sklearn.preprocessing import RobustScaler
import pickle
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--props',nargs='+',type=str,default=['pCMC', 'AW_ST_CMC', 'Area_min'])
    parser.add_argument('--df_path', type=str, default='../data/surfpro_imputed_non_ionic.csv')
    parser.add_argument('--save_folder', type=str, default='models')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--n_epoch', type=int, default=10000)

    return parser.parse_args()


def get_y(df, args):
    df = df.dropna(subset=args.props).reset_index(drop=True)
    df.loc[:, 'Gamma_max'] = df.loc[:,'Gamma_max']*1e6
    
    y_final = [df[prop].to_numpy()[:, None] for prop in args.props]
    y = np.concatenate(y_final, axis=-1)
    # print(y.shape)

    X = df['SMILES'].to_numpy()
    return X,y
    
def get_data(args):
    print(args.df_path)
    df = pd.read_csv(args.df_path)
    
    df_train = df[df['fold']!=-1]
    X_train,y_train = get_y(df_train, args)
    
    df_test = df[df['fold']==-1]
    X_test,y_test = get_y(df_test, args)
    

    X_train_all = X_train
    y_train_all = y_train

    

    

    print(f'Train: {X_train_all.shape}, {y_train_all.shape}')
    print(f'Test: {X_test.shape}, {y_test.shape}')
    

    return (X_train_all.tolist(), y_train_all), (X_test.tolist(), y_test)

    
def get_model(args):
    task = ['regression' for _ in range(len(args.props))]
    print(task)
    
    model = torch_molecule.generator.graph_dit.GraphDITMolecularGenerator(
        device='cuda:0', batch_size=args.batch_size,
        task_type=['regression' for _ in range(len(args.props))],
        epochs=args.n_epoch, verbose=True, 
    )

    return model


def main():
    args = parse_args()
    os.makedirs(args.save_folder, exist_ok=True)
    (train), (test) = get_data(args)
    model = get_model(args)


    model.device = 'cuda'
    model = model.fit(train[0], train[1])

    torch.cuda.empty_cache()
    torch.save(model, f'{args.save_folder}/model_{args.props}.pth')