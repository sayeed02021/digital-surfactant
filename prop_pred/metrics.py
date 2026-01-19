from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')   # disable all RDKit warnings
import warnings
warnings.filterwarnings(action="ignore")


import numpy as np
from rdkit import Chem
import pandas as pd
import os
from sklearn.preprocessing import RobustScaler
import glob
import time
import argparse

from fcd_torch import FCD
from moses.metrics.metrics import internal_diversity, FragMetric

class Metrics(object):
    def __init__(self, path, train_path, method_Name, props,
                 batch_size=1, device='cpu', args=None):
        # self.df = pd.read_csv(path)
        self.df = path
        self.df = self.df.dropna(subset=['gen_smiles']).reset_index(drop=True)
        # self.df = self.df[self.df['split']=='test'].reset_index(drop=True)
        self.gen = self.df['gen_smiles'].to_numpy()
        
        # self.ref = self.df['target_smiles'].to_numpy()
        self.batch_size = batch_size
        self.device = device
        self.train_df = pd.read_csv(train_path)
        # self.train_df z= self.train_df.dropna(subset=props).reset_index(drop=True)
        if args.mol_type=='non-ionic':
            self.train_df = self.train_df[
                                        self.train_df['type'].isin(["non-ionic", "sugar-based non-ionic"])
                                      ].reset_index(drop=True)
        elif args.mol_type=='ionic':
            self.train_df = self.train_df[
                                        ~self.train_df['type'].isin(["non-ionic", "sugar-based non-ionic"])
                                      ].reset_index(drop=True)
            
        
        self.train_df = self.train_df[self.train_df['fold']!=-1].reset_index(drop=True)
        self.ref = self.train_df['SMILES'].to_numpy()
        print('Train df size: ', self.train_df.shape)
        self.name = method_Name
        self.props = props
        self.reset()

    def reset(self):
        self.fcd =  FCD(device=self.device, n_jobs=4, 
                        batch_size=self.batch_size)        
        self.frag_met = FragMetric(n_jobs=4, device=self.device, 
                                   batch_size=self.batch_size)
        

    def calc_dist(self):
        return self.fcd(ref=self.ref, gen=self.gen)


    def calc_div(self):
        return internal_diversity(self.gen, n_jobs=4, device=self.device)
    
    def calc_sim(self):
        return self.frag_met(ref=self.ref, gen=self.gen)
    
    def __call__(self):
        dist = self.calc_dist()
        div = self.calc_div()
        sim = self.calc_sim()
        unique_atoms, unique_atoms_gen = self.coverage()
        mae = self.avg_mae()

        print('Unique atoms in training set: ', unique_atoms)
        print('Generated Unique atoms: ', unique_atoms_gen)

        return {
            'Method': self.name,
            'Diversity': round(div,4),
            'Similarity': round(sim,4),
            'Distance': round(dist,4),
            'Coverage': f'{len(unique_atoms_gen)}/{len(unique_atoms)}',
            'Avg_MAE': mae
        }
    
    def avg_mae(self):
        
        target_props = [f'target_{prop}' for prop in self.props]
        pred_props = [f'pred_{prop}' for prop in self.props]
        target = self.df.loc[:,target_props].to_numpy()
        pred = self.df.loc[:, pred_props].to_numpy()
        mae = np.abs(target-pred)
        return mae.mean(axis=0).tolist()
    
    
    def coverage(self):
        smiles = self.train_df['SMILES'].to_numpy()

        unique_atoms = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            
            for atom in mol.GetAtoms():
                atom_symbol = atom.GetSymbol()
                if atom_symbol not in unique_atoms:
                    unique_atoms.append(atom_symbol)

        unique_atoms_gen = []
        for smi in self.gen:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            for atom in mol.GetAtoms():
                atom_symbol = atom.GetSymbol()
                if atom_symbol not in unique_atoms_gen:
                    unique_atoms_gen.append(atom_symbol)

        return unique_atoms, unique_atoms_gen

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, choices=['trfm', 'diff'], required=True, help='Which method was used to generated molecules')
    parser.add_argument(
        '--pred_folder', type=str, help='Path to folder that contains predicted molecules'
    )
    parser.add_argument(
        '--props', nargs='+', type=str, help='pCMC or pCMC AW_ST_CMC Area_min', required=True
    )
    parser.add_argument(
        '--train_path', type=str, required=True, help='Path to training dataset(.csv file)'
    )
    parser.add_argument(
        '--mol_type', type=str, choices=['non-ionic', 'ionic'], required=True, help='If you are generating non-ionic or ionic molecules'
    )

    return parser.parse_args()



if __name__=='__main__':
    # method = 'trfm'
    # # method = '10K_diff'
    # root_path = 'trfm_gen_mols_final'
    # train_path = 'surfpro_imputed_with_folds.csv'
    # props = ['pCMC', 'AW_ST_CMC', 'Area_min']
    
    args = parse_args()
    props = args.props
    method = args.method
    root_path = args.pred_folder
    train_path = args.train_path

    if len(props)>1:
        name = f'{method}_multi_' + '_'.join(str(p) for p in props) 
        
        gen_df_paths = glob.glob(f'{root_path}/predicted_smiles_*.csv')
        gen_df_list = []
        for path in gen_df_paths:
            gen_df_list.append(pd.read_csv(path))
        final_df = pd.concat(gen_df_list)
        print(final_df.shape)
    else:
        name = f'{method}_single_' + '_'.join(str(p) for p in props) 
        
        gen_df_paths = glob.glob(f'{root_path}/predicted_smiles_*.csv')
        gen_df_list = []
        for path in gen_df_paths:
            gen_df_list.append(pd.read_csv(path))
        final_df = pd.concat(gen_df_list)
        print(final_df.shape)



    start = time.time()
    met = Metrics(
        path=final_df, train_path=train_path, 
        method_Name=name, props=props, args=args
    )



    
    data = met()
    print(data)
    print(f'Calculated in {time.time()-start:0.3f} seconds')
    if os.path.exists("generated_mol_metrics_final.csv"):
        df = pd.DataFrame([data])
        exist_df = pd.read_csv('generated_mol_metrics_final.csv')
        final_df = pd.concat([exist_df, df])
        final_df.to_csv('generated_mol_metrics_final.csv', index=False)
    else:
        df = pd.DataFrame(data)
        df.to_csv('generated_mol_metrics_final.csv', index=False)

