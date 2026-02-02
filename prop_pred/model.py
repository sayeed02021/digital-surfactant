import torch
import torch.nn as nn
import torch_geometric.nn as nng
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
class RegressionHead(nn.Module):
    def __init__(self, n_prop, dim):
        super().__init__()
        self.dim = dim

        self.layers = nn.Sequential(
            nn.LayerNorm(normalized_shape=self.dim),
            nn.Linear(self.dim, 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,n_prop)
        )

    def forward(self, x):
        return self.layers(x)
    



class AttentiveFPModel(nn.Module):
    def __init__(
            self, 
            props,
            args
    ):
        super().__init__()
        self.props = props
        self.n_prop = len(props)
        self.hidden_channels = args.hidden_channels
        self.out_channels = args.out_channels
        self.num_layers = args.num_layers
        self.num_timesteps = args.num_timesteps
        self.dropout = args.dropout
        self.in_channels = args.in_channels
        self.edge_dim = args.edge_dim

        self.encoder = nng.AttentiveFP(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            edge_dim=self.edge_dim,
            num_layers=self.num_layers,
            num_timesteps=self.num_timesteps,
            dropout=self.dropout
        )



        self.head = RegressionHead(
            n_prop=self.n_prop,
            dim = self.out_channels
        )

        self.flat = nn.Flatten()
        

    def forward(self, feats, apply_head = True):
        
        x = self.encoder(
            feats.x,
            feats.edge_index, 
            feats.edge_attr,
            feats.batch
        )

        if apply_head:
            preds = self.head(x)
            
        else:
            preds = x
        return preds
        

class LitAttentiveFP(pl.LightningModule):
    def __init__(self, props, args, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = AttentiveFPModel(props,args)
        self.props = props
        self.criterion = nn.HuberLoss()
        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, feats):
        return self.model(feats)
    
    def training_step(self, batch, batch_idx):
        preds = self(batch)
        targets = batch['y']
        mask = batch['mask']
        masked_preds = preds*mask
        masked_targets = targets*mask
        loss = self.criterion(masked_preds,  masked_targets)
        metrics = self.calc_errors(preds, targets,mask, prefix='train')
        metrics['loss'] = loss
        return metrics

    
    def validation_step(self, batch, batch_idx):
        preds = self(batch)
        targets = batch['y']
        mask = batch['mask']
        metrics = self.calc_errors(preds, targets, mask, prefix='val')
        return metrics

    def test_step(self, batch, batch_idx):
        preds = self(batch)
        targets = batch['y']
        mask = batch['mask']
        metrics = self.calc_errors(preds, targets, mask, prefix='test')
        return metrics
    

    def predict_step(self, batch, batch_idx):
        with torch.set_grad_enabled(False):
            preds = self.forward(batch)
        targets = batch['y']
        smiles = batch['smile']
        mask = batch['mask']
        # if smi_act[0]=='No_Smile': ## used when training
        if batch.source_data_exists[0]: ## target smiles and some other meta data exist
            results = {
                "index": batch['index'].numpy(),
                "smiles": np.array(smiles), 
                "preds": preds.numpy(),
                "targets": targets.numpy(),
                "mask": mask.numpy()
            }
            for key in batch.source_data_keys[0]:
                results[key] = batch[key]

            return results

        else:
            return {
                "index": batch['index'].numpy(),
                "smiles": np.array(smiles),
                "targets": targets.numpy(),
                "preds": preds.numpy(),
                "mask": mask.numpy()
            }
            
				



    def calc_errors(self, preds, targets, mask, prefix):
        """
        targets and preds shape: Nxn_prop
        """
        masked_preds = preds[mask]
        masked_targets = targets[mask]
        metrics = {}

        metrics[f'{prefix}_mae'] = self.mae(masked_preds, masked_targets)
        metrics[f'{prefix}_rmse'] = torch.sqrt(self.mse(masked_preds, masked_targets))

        ## Property-wise metrics:
        if len(self.props)>1:
            for i, prop in enumerate(self.props):
                prop_mask = mask[:,i]
                if prop_mask.sum()==0:
                    continue
                metrics[f'{prefix}_{prop}_mae'] = self.mae(targets[:,i][prop_mask], preds[:,i][prop_mask])
                metrics[f'{prefix}_{prop}_rmse'] = torch.sqrt(self.mse(targets[:,i][prop_mask], preds[:,i][prop_mask]))

        else:
            metrics[f'{prefix}_{self.props[0]}_mae'] = metrics[f'{prefix}_mae']
            metrics[f'{prefix}_{self.props[0]}_rmse'] = metrics[f'{prefix}_rmse']

        self.log_dict(metrics, on_epoch=True,on_step=False, batch_size=64)

        return metrics

    

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr) 
    


