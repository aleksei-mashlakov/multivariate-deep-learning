"""
Runs a model on a single node on 4 GPUs.
"""
import os
import csv
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import torch

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TestTubeLogger
from datetime import datetime
from argparse import ArgumentParser
from model import DSANet
from datautil import DataUtil

SEED = 7
torch.manual_seed(SEED)
np.random.seed(SEED)
val_results = pd.DataFrame()
test_results = pd.DataFrame()

def main(hparams):
    """
    Main training routine specific for this project
    """
    global val_results, test_results
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    print('loading model...')
    model = DSANet(hparams)
    print('model built')
    print(f"\n#######\nTESTING hparams: mv:{hparams.n_multiv}, bs:{hparams.batch_size}, drop:{hparams.drop_prob}, lr:{hparams.learning_rate}, d_model:{hparams.d_model}, local:{hparams.local}, n_kernels:{hparams.n_kernels}, window:{hparams.window}\n#######")
   
    # ------------------------
    # 2 INIT TEST TUBE EXP
    # ------------------------
    filename = '{}{}{}{}{}{}'.format('my_dsanet_', hparams.data_name ,'_', hparams.powerset, '_', str(hparams.calendar)) 
    logger = TestTubeLogger("tb_logs_v2", filename)
    # ------------------------
    # 3 DEFINE CALLBACKS
    # ------------------------
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=35,
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
        fast_dev_run=False
    )
    # ------------------------
    # 5 START TRAINING
    # ------------------------
    if not hparams.test_only:
        st_time = datetime.now()
        result = trainer.fit(model)
        eval_result = model.val_results
        df1=pd.DataFrame(eval_result, [0])
        #print(result)
        eval_time = str(datetime.now() - st_time)
        print(f"Train time: {eval_time}, Results: {eval_result}")

        st_time = datetime.now()
        model.hparams.mcdropout = 'True'
        trainer.test(model)
        eval_time = str(datetime.now() - st_time)
        test_result = model.test_results
        df2 = pd.DataFrame(test_result, [0])
        print(f"Test time: {eval_time}, Results: {test_result}")

        df1 = pd.concat([df1, pd.DataFrame(vars(hparams), [0])], axis=1, sort=False)
        df2 = pd.concat([df2, pd.DataFrame(vars(hparams), [0])], axis=1, sort=False)
        val_results = pd.concat([val_results, df1], axis=0, sort=False)
        test_results = pd.concat([test_results, df2], axis=0, sort=False)
        val_filename = '{}{}{}{}{}{}'.format(filename, '_', str(hparams.window), '_', str(hparams.horizon), '_val.csv' ) 
        test_filename = '{}{}{}{}{}{}'.format(filename, '_', str(hparams.window), '_', str(hparams.horizon), '_test.csv' ) 
        val_results.to_csv(val_filename, mode='a')
        test_results.to_csv(test_filename, mode='a')
    else:
        st_time = datetime.now()
        model.hparams.mcdropout = 'True'
        trainer.test(model)
        eval_time = str(datetime.now() - st_time)
        test_result = model.test_results
        df2 = pd.DataFrame(test_result, [0])
        print(f"Test time: {eval_time}, Results: {test_result}")
        df2 = pd.concat([df2, pd.DataFrame(vars(hparams), [0])], axis=1, sort=False)
        test_results = pd.concat([test_results, df2], axis=0, sort=False)
        test_filename = '{}{}{}{}{}{}'.format(filename, '_', str(hparams.window), '_', str(hparams.horizon), '_test.csv' ) 
        test_results.to_csv(test_filename, mode='a')


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
    dataset = DataUtil(hyperparams, 2)
    if hasattr(dataset, 'scale'):
        print('we have scale')
        setattr(hyperparams, 'scale', dataset.scale)
        print(dataset.scale)
    if hasattr(dataset, 'scaler'):
        print('we have scaler')
        setattr(hyperparams, 'scaler', dataset.scaler)
        print(dataset.scaler)

    setattr(hyperparams, 'n_multiv', dataset.m)
    print(f'RUNNING ON GPU')
    main(hyperparams)