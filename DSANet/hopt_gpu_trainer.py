"""
Runs a model on a single node on CPU only.
"""
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import traceback

from test_tube import HyperOptArgumentParser, Experiment
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import LightningModule

import _pickle as pickle
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt import base
import hyperopt
import csv

from model import DSANet

SEED = 7
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)


def optimize(optimizer_params):
    """
    Main training routine specific for this project
    """
    logging.basicConfig(level=logging.INFO)
    # dirs
    root_dir = os.path.dirname(os.path.realpath(__file__))
    demo_log_dir = os.path.join(root_dir, 'dsanet_logs')
    checkpoint_dir = os.path.join(demo_log_dir, 'model_weights')
    test_tube_dir = os.path.join(demo_log_dir, 'test_tube_data')

    # although we user hyperOptParser, we are using it only as argparse right now
    parent_parser = HyperOptArgumentParser(strategy='grid_search', add_help=False)

    # gpu args
    parent_parser.add_argument('--test_tube_save_path', type=str, default=test_tube_dir, help='where to save logs')
    parent_parser.add_argument('--model_save_path', type=str, default=checkpoint_dir, help='where to save model')

    # allow model to overwrite or extend args
    parser = DSANet.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()
    print(hyperparams)
    setattr(hyperparams, 'batch_size', int(optimizer_params['batch_size']))
    setattr(hyperparams, 'drop_prob', optimizer_params['dropout'])
    setattr(hyperparams, 'learning_rate', optimizer_params['learning_rate'])
    setattr(hyperparams, 'd_model', int(optimizer_params['units']))
    # hyperparams['batch_size'] = optimizer_params['batch_size']
    # hyperparams['drop_prob'] = optimizer_params['dropout']
    # hyperparams['learning_rate'] = optimizer_params['learning_rate']
    # hyperparams['d_model'] = optimizer_params['units']
    print(hyperparams)
    hparams = hyperparams
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------

    print('loading model...')
    model = DSANet(hparams)
    print('model built')

    # ------------------------
    # 2 INIT TEST TUBE EXP
    # ------------------------

    # init experiment
    exp = Experiment(
        name='dsanet_exp_{}_window={}_horizon={}'.format(hparams.data_name, hparams.window, hparams.horizon),
        save_dir=hparams.test_tube_save_path,
        autosave=False,
        description='test demo'
    )

    exp.argparse(hparams)
    exp.save()

    # ------------------------
    # 3 DEFINE CALLBACKS
    # ------------------------
    model_save_path = '{}/{}/{}'.format(hparams.model_save_path, exp.name, exp.version)

    checkpoint_callback = ModelCheckpoint(
        filepath=model_save_path,
        save_best_only=True,
        verbose=True,
        monitor='val_loss',
        mode='auto'
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=25,
        verbose=True,
        mode='min'
    )

    # ------------------------
    # 4 INIT TRAINER
    # ------------------------
    trainer = Trainer(
        gpus="0,1",
        distributed_backend='ddp',
        experiment=exp,
        early_stop_callback=early_stop,
        checkpoint_callback=checkpoint_callback,
    )

    # ------------------------
    # 5 START TRAINING
    # ------------------------
    st_time = datetime.now()
    trainer.fit(model)
    eval_time = str(datetime.now() - st_time)
    print("Iteration %d: Getting results ... " % ITERATION)
    csv_load_path = '{}/{}/{}{}'.format(hparams.test_tube_save_path, exp.name, 'version_', exp.version)
    df = pd.read_csv('{}/{}'.format(csv_load_path, 'metrics.csv'))  # change to experiment save dir
    min_idx = df['val_nd'].idxmin()

    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([optimizer_params, hparams, df['tng_loss'].iloc[min_idx], df['val_loss'].iloc[min_idx],
                     df['val_nd'].iloc[min_idx], df['NRMSE'].iloc[min_idx], df['val_rho10'].iloc[min_idx],
                     df['val_rho50'].iloc[min_idx], df['val_rho90'].iloc[min_idx], eval_time, STATUS_OK])
    of_connection.close()

    return {'loss': df['val_nd'].iloc[min_idx],
            'ND': df['val_nd'].iloc[min_idx],
            'NRMSE': df['NRMSE'].iloc[min_idx],
            'val_loss': df['val_loss'].iloc[min_idx],
            'params': optimizer_params,
            'rho_metric': {'rho10': df['val_rho10'].iloc[min_idx], 'rho50': df['val_rho50'].iloc[min_idx],
                           'rho90': df['val_rho90'].iloc[min_idx]},
            'iteration': ITERATION,
            'eval_time': eval_time,
            'status': STATUS_OK}
    # trainer.test()
    # print('View tensorboard logs by running\ntensorboard --logdir %s' % os.getcwd())
    # print('and going to http://localhost:6006 on your browser')


if __name__ == '__main__':

    # !NOTE this out_file should be updated during the training
    global out_file
    out_file = os.path.join('./save', 'train_results.csv')
    results_pickle_file = os.path.join('./save', 'results_pickle.pkl')
    nb_evals = 100
    if not os.path.isfile(out_file):
        of_connection = open(out_file, 'w')
        writer = csv.writer(of_connection)
        # Write the headers to the file
        writer.writerow(
            ['params', 'hparams', 'train_loss', 'valid_loss', 'val_ND', 'val_NRMSE', 'val_rho10', 'val_rho50',
             'val_rho90', 'time', 'status'])
        of_connection.close()

    # Global variable
    global ITERATION

    print("Attempt to resume a past training if it exists:")
    try:
        # https://github.com/hyperopt/hyperopt/issues/267
        trials = pickle.load(open(results_pickle_file, "rb"))
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
                     'batch_size': 2 ** hp.quniform('batch_size', 6, 10, 1),
                     'units': hp.choice('units', [2 ** hp.quniform('units_a', 5, 8, 1),
                                                  25 * (2 ** hp.quniform('units_b', 0, 3, 1))]),
                     'learning_rate': hp.choice('learning_rate', [5 * 10 ** -hp.quniform('lr_a', 3, 4, 1),
                                                                  1 * 10 ** -hp.quniform('lr_b', 2, 4, 1)])}
    best = fmin(optimize,
                search_params,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials,
                rstate=np.random.RandomState(42),
                show_progressbar=False)
    pickle.dump(trials, open(results_pickle_file, "wb"))
    print('best: ')
    print(best)
    print(hyperopt.space_eval(search_params, best))

    # ---------------------
    # RUN TRAINING
    # ---------------------
    # run on HPC cluster
    # print(f'RUNNING ON CPU')
    # * change the following code to comments for grid search
    # main(hyperparams)

    # * recover the following code for grid search
    # hyperparams.optimize_parallel_gpu(
    #    main,
    #    gpu_ids="0",
    #    max_nb_trials=200,    # this number needs to be adjusted according to the actual situation
    # )
