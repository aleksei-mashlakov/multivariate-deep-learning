"""
Runs a model on a single node on 4 GPUs.
"""
import os
from argparse import ArgumentParser
import pytorch_lightning as pl

import numpy as np
import pandas as pd
from datetime import datetime

import torch

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TestTubeLogger

import csv

from model import DSANet

SEED = 7
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(hparams):
    """
    Main training routine specific for this project
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------

    print('loading model...')
    model = DSANet(hparams)
    print('model built')

    # ------------------------
    # 2 INIT TEST TUBE EXP
    # ------------------------
    logger = TestTubeLogger("tb_logs_v2", name="my_dsanet_pow")

    # ------------------------
    # 3 DEFINE CALLBACKS
    # ------------------------
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=25,
        verbose=False,
        mode='min'
    )

    # ------------------------
    # 4 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        gpus=2,
        distributed_backend='dp',
        logger=logger,
        early_stop_callback=early_stop_callback,
        show_progress_bar=False,
        profiler=True,
    )

    # ------------------------
    # 5 START TRAINING
    # ------------------------
    st_time = datetime.now()
    result = trainer.fit(model)
    print(result)
    eval_time = str(datetime.now() - st_time)
    print(f"Train time: {eval_time}")
    
    st_time = datetime.now()
    result = trainer.test()
    eval_time = str(datetime.now() - st_time)
    print(f"Test time: {eval_time}")
    print(result)

if __name__ == '__main__':

    root_dir = os.path.dirname(os.path.realpath(__file__))
    demo_log_dir = os.path.join(root_dir, 'dsanet_logs')
    checkpoint_dir = os.path.join(demo_log_dir, 'model_weights')
    test_tube_dir = os.path.join(demo_log_dir, 'test_tube_data')

    # although we user hyperOptParser, we are using it only as argparse right now
    parent_parser = ArgumentParser( add_help=False)

    # gpu args
    parent_parser.add_argument('--test_tube_save_path', type=str, default=test_tube_dir, help='where to save logs')
    parent_parser.add_argument('--model_save_path', type=str, default=checkpoint_dir, help='where to save model')

    # allow model to overwrite or extend args
    parser = DSANet.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()

    print(f'RUNNING ON GPU')
    main(hyperparams)