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

import _pickle as pickle
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import csv

SEED = 7
torch.manual_seed(SEED)
np.random.seed(SEED)

val_results = pd.DataFrame()
test_results = pd.DataFrame()

def optimize(optimizer_params):
    """
    Main training routine specific for this project
    """
    global val_results, test_results
    global val_out_file, test_out_file, ITERATION, epochs
    ITERATION += 1
    root_dir = os.path.dirname(os.path.realpath(__file__))
    # although we user hyperOptParser, we are using it only as argparse right now
    parent_parser = ArgumentParser( add_help=False)

    # allow model to overwrite or extend args
    parser = DSANet.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()
    dataset = DataUtil(hyperparams, 2)
    if hasattr(dataset, 'scale'):
        #print('we have scale')
        setattr(hyperparams, 'scale', dataset.scale)
        #print(dataset.scale)
    if hasattr(dataset, 'scaler'):
        #print('we have scaler')
        setattr(hyperparams, 'scaler', dataset.scaler)
        #rint(dataset.scaler)

    setattr(hyperparams, 'n_multiv', dataset.m)
    setattr(hyperparams, 'batch_size', int(optimizer_params['batch_size']))
    setattr(hyperparams, 'drop_prob', optimizer_params['dropout'])
    setattr(hyperparams, 'learning_rate', optimizer_params['learning_rate'])
    setattr(hyperparams, 'd_model', int(optimizer_params['units']))
    setattr(hyperparams, 'local', int(optimizer_params['local']))
    setattr(hyperparams, 'n_kernels', int(optimizer_params['n_kernels']))
    setattr(hyperparams, 'window', int(optimizer_params['window']))
    hparams = hyperparams
    print(f"\n#######\nTESTING hparams: mv:{hparams.n_multiv}, bs:{hparams.batch_size}, drop:{hparams.drop_prob}, lr:{hparams.learning_rate}, d_model:{hparams.d_model}, local:{hparams.local}, n_kernels:{hparams.n_kernels}, window:{hparams.window}\n#######")
    
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    print('loading model...')
    model = DSANet(hparams)
    print('model built')
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
        patience=5,
        verbose=False,
        mode='min'
    )
    # ------------------------
    # 4 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        gpus=4,
        distributed_backend='dp',
        logger=logger,
        early_stop_callback=early_stop_callback,
        show_progress_bar=False,
        profiler=True,
        fast_dev_run=False,
        max_epochs=100
    )
    # ------------------------
    # 5 START TRAINING
    # ------------------------
    st_time = datetime.now()
    result = trainer.fit(model)
    eval_result = model.val_results
    df1=pd.DataFrame(eval_result, [ITERATION])
    print(result)
    eval_time = str(datetime.now() - st_time)
    print(f"Train time: {eval_time}, Results: {eval_result}")

    st_time = datetime.now()
    model.hparams.mcdropout = 'True'
    trainer.test(model)
    eval_time = str(datetime.now() - st_time)
    test_result = model.test_results
    df2 = pd.DataFrame(test_result, [ITERATION]) 
    print(f"Test time: {eval_time}, Results: {test_result}")
    df1 = pd.concat([df1, pd.DataFrame(vars(hparams), [ITERATION])], axis=1, sort=False)
    df2 = pd.concat([df2, pd.DataFrame(vars(hparams), [ITERATION])], axis=1, sort=False)

    val_results = pd.concat([val_results, df1], axis=0, sort=False)
    test_results = pd.concat([test_results, df2], axis=0, sort=False)
    return eval_result['val_nd_all']

if __name__ == '__main__':
    # !NOTE this out_file should be updated during the training
    global val_out_file, test_out_file, ITERATION, epochs
    
    val_out_file = os.path.join('./save', 'val_results_electricity_full.csv')
    test_out_file = os.path.join('./save', 'test_results_electricity_full.csv')
    results_pickle_file = os.path.join('./save', 'train_results_electricity_pickle_full.pkl')
    nb_evals = 25
    max_evals = 1

    print("Attempt to resume a past training if it exists:")
    while max_evals <= 150:
        try:
            # https://github.com/hyperopt/hyperopt/issues/267
            trials = pickle.load(open(results_pickle_file, "rb"))
            val_results = pd.read_csv(val_out_file, index_col=0)
            test_results = pd.read_csv(test_out_file, index_col=0)
            print("Found saved Trials! Loading...")
            max_evals = nb_evals + len(trials.trials)
            ITERATION = len(trials.trials)
            print("Rerunning from %d trials to add another one." % len(trials.trials))
        except:
            trials = Trials()
            max_evals = nb_evals
            ITERATION = 0
            print("Starting from scratch: new trials.")

        search_params = {'dropout': hp.quniform('dropout', 0.1, 0.5, 0.1),
                         'batch_size': 2 ** hp.quniform('batch_size', 4, 8, 1),
                         'units': hp.choice('units', [2 ** hp.quniform('units_a', 5, 10, 1),
                                                      25 * (2 ** hp.quniform('units_b', 0, 4, 1))]),
                         'learning_rate': hp.choice('learning_rate', [5 * 10 ** -hp.quniform('lr_a', 3, 4, 1),
                                                                      1 * 10 ** -hp.quniform('lr_b', 2, 4, 1)]),
                         'local': hp.choice('local', [3, 5, 7]),
                         'n_kernels': hp.choice('n_kernels', [32, 50]), #, 50, 100
                         'window': hp.choice('window', [128, 168])} # 48, 72, 128,
                         
        best = fmin(optimize,
                    search_params,
                    algo=tpe.suggest,
                    max_evals=max_evals,
                    trials=trials,
                    rstate=np.random.RandomState(42),
                    show_progressbar=False)
        pickle.dump(trials, open(results_pickle_file, "wb"))
        val_results.to_csv(val_out_file)
        test_results.to_csv(test_out_file)
        print('best: ')
        print(best)

    print('best: ')
    print(best)
    print('We are done here...')