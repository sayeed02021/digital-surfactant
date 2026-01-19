import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateFinder
from pytorch_lightning.tuner.tuning import Tuner
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from dataset import SMILESDataset
from model import LitAttentiveFP
import pandas as pd
import numpy as np
import os
import yaml
import argparse
from pytorch_lightning import seed_everything
from collections import defaultdict

from utils import ensemble_results




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, help="path to YAML config")
    cmd_args = parser.parse_args()
    with open(cmd_args.config_file, "r") as f:
        config = yaml.safe_load(f)

    return argparse.Namespace(**config)


def get_data_vecs(path,args, mode):
    df = pd.read_csv(path)
    smiles = df['SMILES'].to_numpy()
    y_val = df[args.props].to_numpy()
    indices = np.arange(len(df))
    if mode=='train':
        folds = df['fold'].to_numpy()
    else:
        folds = -1*np.ones(len(df)).tolist()
    return smiles, y_val, indices, folds



def main():
    # seed_everything()
    
    args = parse_args()
    if len(args.props)==1:
        args.scaling=False
    props = args.props
    # print(args.save_path)
    seed_everything(seed=args.SEED, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.makedirs(args.save_path, exist_ok=True)

    loss_data = defaultdict(list)

    train_path = '../data/surfpro_train.csv'
    test_path = '../data/surfpro_test.csv'
    train_smiles, train_y, train_index, folds = get_data_vecs(train_path, args, mode='train')
    # train_dataset = SMILESDataset(
    #     root=args.save_path,
    #     csv_path='../data/surfpro_train.csv',
    #     feat = args.props,
    #     mode='train',
    #     scale_data=args.scaling
    # )

    train_dataset = SMILESDataset(
        # root=args.save_path,
        smiles=train_smiles,
        index=train_index,
        mode='train',
        y_val=train_y,
        feat=args.props,
        scale_data=args.scaling,
        fold_values=folds

    )

    if args.scaling:
        print('SCALING DATA')
    else:
        print("NOT SCALING DATA")


    test_smiles, test_y, test_index, folds = get_data_vecs(test_path, args, mode='test')
    # test_dataset = SMILESDataset(
    #     root=args.save_path,
    #     csv_path='../data/surfpro_test.csv',
    #     mode='test',
    #     feat = args.props,
    #     scaler=train_dataset.scaler,
    #     scale_data = args.scaling
    # )

    test_dataset = SMILESDataset(
        # root=args.save_path,
        smiles=test_smiles,
        index=test_index,
        mode='test',
        y_val=test_y,
        feat=args.props,
        scale_data=args.scaling,
        scaler = train_dataset.scaler

    )
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False)
    
    all_preds = defaultdict(list)


    for fold in range(args.n_folds):
        print('FOLD: ', fold)

        train_idx = [i for i,d in enumerate(train_dataset) if d.fold!=fold]
        val_idx = [i for i,d in enumerate(train_dataset) if d.fold==fold]
        # train_fold = train_dataset[train_idx]
        # val_fold = train_dataset[val_idx]    
        train_fold = Subset(train_dataset, train_idx)
        val_fold = Subset(train_dataset, val_idx) 
        print(f'Train: {len(train_fold)}, Val: {len(val_fold)}')
        train_loader = DataLoader(train_fold, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(val_fold, batch_size=args.batch_size, shuffle=False)
        print(len(train_loader), len(valid_loader), len(test_loader))
        model = LitAttentiveFP(props=props, args=args)
        # checkpoint = ModelCheckpoint(monitor='val_mae', save_top_k=1, mode='min')
        
        callbacks = []
        if args.tuning:
            callbacks = [LearningRateFinder()]

        if args.stop_early:
            callbacks +=[EarlyStopping(
                monitor='val_mae',
                patience=50,
                mode='min',
                check_on_train_epoch_end=False
            )]
        


        trainer = pl.Trainer(
            max_epochs=args.num_epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            callbacks=callbacks, 
            logger = None,
            precision=32,
            default_root_dir=args.save_path,
            enable_model_summary=False
        )

        
        

        
        trainer.fit(model, train_loader, valid_loader)
        model.save_hyperparameters()
        print('USED LEARNING RATE: ', model.hparams.lr)
        args.lr = model.hparams.lr
        with open(f"{args.save_path}/lightning_logs/version_{fold}/config.yaml", "w") as f:
            yaml.dump(vars(args), f, default_flow_style=False)
        val_metrics = trainer.validate(model, valid_loader, verbose=False)[0]
        test_metrics = trainer.test(model, test_loader, verbose=False)[0]
        loss_data['fold'].append(fold)
        best_ep = trainer.current_epoch
        loss_data['epoch'].append(best_ep)
        for prop in props:
            # print(val_metrics, test_metrics)
            for key, val in val_metrics.items():
                if prop in key:
                    loss_data[key].append(val)
            for key, val in test_metrics.items():
                if prop in key:
                    loss_data[key].append(val)


                

        # print(len(final_data))
        # torch.save(model.state_dict(), f'{args.save_path}/Fold_{fold}.pt')    
        loss_df = pd.DataFrame(loss_data)
        loss_df.to_csv(f'{args.save_path}/metrics.csv', index=False)

        
        

        fold_outputs = trainer.predict(model, test_loader)
        for out in fold_outputs:
            smiles = out['smiles']
            targets = out['targets'].cpu().numpy()
            preds = out['preds'].cpu().numpy()
            masks = out['mask'].cpu().numpy()
            # Un-scale predictions and targets for properties where a value exists
            # preds_unscaled = np.full_like(preds, np.nan)
            # targets_unscaled = np.full_like(targets, np.nan)

            # Apply the scaler only to the values that are not masked
            if args.scaling:
                print('Scaling')
                preds_unscaled = test_dataset.scaler.inverse_transform(preds)
                targets_unscaled = test_dataset.scaler.inverse_transform(targets)
            else:

                preds_unscaled = preds
                targets_unscaled = targets
            preds_unscaled[~masks] = np.nan
            targets_unscaled[~masks] = np.nan
            
            for i in range(len(smiles)):
                s = smiles[i]
                all_preds["smiles"].append(s)
                all_preds["fold"].append(fold)
                for j, prop in enumerate(props):
                    all_preds[f"{prop}_target"].append(targets_unscaled[i, j])
                    all_preds[f"{prop}_pred"].append(preds_unscaled[i, j])

                
    all_preds = pd.DataFrame(all_preds)
    all_preds.to_csv(f'{args.save_path}/all_preds.csv', index=False)


    ensemble_results(all_preds, props, args)

        # trainer.test(model, test_loader)

if __name__=="__main__":
    main()

        