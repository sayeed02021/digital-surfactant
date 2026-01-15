import argparse

import configuration.opts as opts
from trainer_2.transformer_trainer import TransformerTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.train_opts(parser)
    opt = parser.parse_args()
    print(opt)

    trainer = TransformerTrainer(opt)
    trainer.train(opt)
