import os
import pickle as pkl

import torch

import configuration.config_default as cfgd
import utils.log as ul
import utils.file as uf
import utils.torch_util as ut
import preprocess.vocabulary as mv
from models.transformer.encode_decode.model import EncoderDecoder
from models.transformer.module.noam_opt import NoamOpt as moptim
from models.transformer.module.decode import decode
from trainer_2.base_trainer import BaseTrainer
from models.transformer.module.label_smoothing import LabelSmoothing
from models.transformer.module.simpleloss_compute import SimpleLossCompute
import matplotlib.pyplot as plt 

class TransformerTrainer(BaseTrainer):

    def __init__(self, opt):
        super().__init__(opt)

    def get_model(self, opt, vocab, device):
        vocab_size = len(vocab.tokens())
        if opt.starting_epoch == 1:
            model = EncoderDecoder.make_model(
                vocab_size, vocab_size, N=opt.N,
                d_model=opt.d_model, d_ff=opt.d_ff, h=opt.H, dropout=opt.dropout)
        else:
            file_name = os.path.join(self.save_path, f'checkpoint/model_{opt.starting_epoch-1}.pt')
            model = EncoderDecoder.load_from_file(file_name)
        model.to(device)
        return model

    def _initialize_optimizer(self, model, opt):
        optim = moptim(
            model.src_embed[0].d_model, opt.factor, opt.warmup_steps,
            torch.optim.Adam(model.parameters(), lr=0, betas=(opt.adam_beta1, opt.adam_beta2), eps=opt.adam_eps)
        )
        return optim

    def _load_optimizer_from_epoch(self, model, file_name):
        checkpoint = torch.load(file_name, map_location='cuda:1')
        optim_dict = checkpoint['optimizer_state_dict']
        optim = moptim(
            optim_dict['model_size'], optim_dict['factor'], optim_dict['warmup'],
            torch.optim.Adam(model.parameters(), lr=0)
        )
        optim.load_state_dict(optim_dict)
        return optim

    def get_optimization(self, model, opt):
        if opt.starting_epoch == 1:
            return self._initialize_optimizer(model, opt)
        else:
            file_name = os.path.join(self.save_path, f'checkpoint/model_{opt.starting_epoch-1}.pt')
            return self._load_optimizer_from_epoch(model, file_name)

    def train_epoch(self, dataloader, model, loss_compute, device):
        pad = cfgd.DATA_DEFAULT['padding_value']
        total_loss = 0
        total_tokens = 0
        for i, batch in enumerate(ul.progress_bar(dataloader, total=len(dataloader))):
            src, source_length, trg, src_mask, trg_mask, _, _ = batch
            trg_y = trg[:, 1:].to(device)
            ntokens = float((trg_y != pad).data.sum())
            src = src.to(device)
            trg = trg[:, :-1].to(device)
            src_mask = src_mask.to(device)
            trg_mask = trg_mask.to(device)
            out = model.forward(src, trg, src_mask, trg_mask)
            loss = loss_compute(out, trg_y, ntokens)
            total_tokens += ntokens
            total_loss += float(loss)
        loss_epoch = total_loss / total_tokens
        return loss_epoch

    def validation_stat(self, dataloader, model, loss_compute, device, vocab):
        pad = cfgd.DATA_DEFAULT['padding_value']
        total_loss = 0
        n_correct = 0
        total_n_trg = 0
        total_tokens = 0
        tokenizer = mv.SMILESTokenizer()
        for i, batch in enumerate(ul.progress_bar(dataloader, total=len(dataloader))):
            src, source_length, trg, src_mask, trg_mask, _, _ = batch
            trg_y = trg[:, 1:].to(device)
            ntokens = float((trg_y != pad).data.sum())
            src = src.to(device)
            trg = trg[:, :-1].to(device)
            src_mask = src_mask.to(device)
            trg_mask = trg_mask.to(device)
            with torch.no_grad():
                out = model.forward(src, trg, src_mask, trg_mask)
                loss = loss_compute(out, trg_y, ntokens)
                total_loss += float(loss)
                total_tokens += ntokens
                max_length_target = cfgd.DATA_DEFAULT['max_sequence_length']
                smiles = decode(model, src, src_mask, max_length_target, type='greedy')
                for j in range(trg.size()[0]):
                    seq = smiles[j, :]
                    target = trg[j]
                    target = tokenizer.untokenize(vocab.decode(target.cpu().numpy()))
                    seq = tokenizer.untokenize(vocab.decode(seq.cpu().numpy()))
                    if seq == target:
                        n_correct += 1
            n_trg = trg.size()[0]
            total_n_trg += n_trg
        accuracy = n_correct * 1.0 / total_n_trg
        loss_epoch = total_loss / total_tokens
        return loss_epoch, accuracy

    def _get_model_parameters(self, vocab_size, opt):
        return {
            'vocab_size': vocab_size,
            'N': opt.N,
            'd_model': opt.d_model,
            'd_ff': opt.d_ff,
            'H': opt.H,
            'dropout': opt.dropout
        }

    def save(self, model, optim, epoch, vocab_size, opt):
        save_dict = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.save_state_dict(),
            'model_parameters': self._get_model_parameters(vocab_size, opt)
        }
        file_name = os.path.join(self.save_path, f'checkpoint/model_{epoch}.pt')
        uf.make_directory(file_name, is_dir=False)
        torch.save(save_dict, file_name)

    





    def train(self, opt, pretrain=False, finetune=False, pretrained_model_path=None):
    # Load vocabulary
        with open(os.path.join(opt.data_path, 'vocab.pkl'), "rb") as input_file:
            vocab = pkl.load(input_file)
            vocab_size = len(vocab.tokens())

    # Dataloaders
        dataloader_train = self.initialize_dataloader(opt.data_path, opt.batch_size, vocab, 'train')
        dataloader_validation = self.initialize_dataloader(opt.data_path, opt.batch_size, vocab, 'validation')
        device = ut.allocate_gpu()

    # Model
        if finetune and pretrained_model_path:
            model = EncoderDecoder.load_from_file(pretrained_model_path)
            model.to(device)
        else:
            model = self.get_model(opt, vocab, device)

    # Optimizer + loss
        optim = self.get_optimization(model, opt)
        pad_idx = cfgd.DATA_DEFAULT['padding_value']
        criterion = LabelSmoothing(size=len(vocab), padding_idx=pad_idx, smoothing=opt.label_smoothing)

    # Tracking variables
        loss_epoch_train_list = []
        loss_epoch_validation_list = []
        best_val_loss = float("inf")
        patience = getattr(opt, "patience", 10)   # default patience=10 if not in args
        patience_counter = 0

        for epoch in range(opt.starting_epoch, opt.starting_epoch + opt.num_epoch):
            self.LOG.info("Starting EPOCH #%d", epoch)

        # TRAIN 
            self.LOG.info("Training start")
            model.train()
            loss_epoch_train = self.train_epoch(
                dataloader_train, model,
                SimpleLossCompute(model.generator, criterion, optim), device)
            loss_epoch_train_list.append(loss_epoch_train)
            self.LOG.info("Training end")

        # current mode : Save every epoch , change accordingly
            if epoch % 1 == 0:
                self.save(model, optim, epoch, vocab_size, opt)

        #  VALIDATION 
            self.LOG.info("Validation start")
            model.eval()
            loss_epoch_validation, accuracy = self.validation_stat(
                dataloader_validation, model,
                SimpleLossCompute(model.generator, criterion, None),
                device, vocab)
            loss_epoch_validation_list.append(loss_epoch_validation)
            self.LOG.info("Validation end")

            self.LOG.info(
                "Train loss, Validation loss, accuracy: {}, {}, {}".format(
                    loss_epoch_train, loss_epoch_validation, accuracy))
            self.to_tensorboard(loss_epoch_train, loss_epoch_validation, accuracy, epoch)

        #  EARLY STOPPING 
            if loss_epoch_validation < best_val_loss:
                best_val_loss = loss_epoch_validation
                patience_counter = 0
                # Save best model
                self.save(model, optim, epoch, vocab_size, opt)
                self.LOG.info(f"Validation improved. Saving model at epoch {epoch}")
            else:
                patience_counter += 1
                self.LOG.info(f"No improvement. Patience counter = {patience_counter}/{patience}")
                if patience_counter >= patience:
                    self.LOG.info("Early stopping triggered!")
                    break

        plt.figure(figsize=(8, 5))
        plt.plot(range(opt.starting_epoch, opt.starting_epoch + len(loss_epoch_train_list)),loss_epoch_train_list, label='Training Loss', marker='o')
        plt.plot(range(opt.starting_epoch, opt.starting_epoch + len(loss_epoch_validation_list)),loss_epoch_validation_list, label='Validation Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title("train_vs_validation_loss_curve")
        plt.legend()
        plt.grid(True)

    # Save inside checkpoints folder
        plots_dir = os.path.join(self.save_path, "checkpoint")
        os.makedirs(plots_dir, exist_ok=True)
        plot_filename = os.path.join(plots_dir, "train_vs_validation_loss_curve.png")
        plt.savefig(plot_filename, dpi=300)
        plt.close()

        self.LOG.info(f"Loss curve saved as {plot_filename}")

