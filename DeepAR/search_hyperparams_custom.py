import os
import sys
import logging
import argparse
import pickle
import multiprocessing
from copy import copy
from itertools import product
from subprocess import check_call

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, STATUS_FAIL
from hyperopt import base
import hyperopt
base.have_bson = False
import numpy as np
import json
import traceback
import numpy as np
import utils
import time

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

from pytorchtools import EarlyStopping
from datetime import datetime
import csv
import model.net as net
from evaluate import evaluate
from dataloader import *
# from train import train_and_evaluate

logger = logging.getLogger('DeepAR.Searcher')

PYTHON = sys.executable
gpu_ids: list
param_template: utils.Params
args: argparse.ArgumentParser

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='elect', help='Dataset name')
parser.add_argument('--data-dir', default='data', help='Directory containing the dataset')
parser.add_argument('--data-folder', default='data', help='Parent dir of the dataset')
parser.add_argument('--model-name', default='param_search', help='Parent directory for all jobs')
parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--gpu-ids', nargs='+', default=[0], type=int, help='GPU ids')
parser.add_argument('--sampling', action='store_true', help='Whether to do ancestral sampling during evaluation')
parser.add_argument('--save-best', action='store_true', help='Whether to save best ND to param_search.txt')
parser.add_argument('--save-file', action='store_true', help='Whether to save during evaluation')
parser.add_argument('--plot-figure', action='store_true', help='Whether to plot figures')  #
parser.add_argument('--evals', type=int, default=100, help='Dimention of hop tests. Default=100')
parser.add_argument('--restore-file', default=None,
                    help='Optional, name of the file in --model_dir containing weights to reload before \
                    training')  #


def train(model: nn.Module,
          optimizer: optim,
          loss_fn,
          train_loader: DataLoader,
          test_loader: DataLoader,
          params: utils.Params,
          sampling: None,
          epoch: int) -> float:
    '''Train the model on one epoch by batches.
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        train_loader: load train data and labels
        test_loader: load test data and labels
        params: (Params) hyperparameters
        epoch: (int) the current training epoch
    '''
    model.train()
    loss_epoch = np.zeros(len(train_loader))
    # Train_loader:
    # train_batch ([batch_size, train_window, 1+cov_dim]): z_{0:T-1} + x_{1:T}, note that z_0 = 0;
    # idx ([batch_size]): one integer denoting the time series id;
    # labels_batch ([batch_size, train_window]): z_{1:T}.
    for i, (train_batch, idx, labels_batch) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        batch_size = train_batch.shape[0]

        train_batch = train_batch.permute(1, 0, 2).to(torch.float32).to(params.device)  # not scaled
        labels_batch = labels_batch.permute(1, 0).to(torch.float32).to(params.device)  # not scaled
        idx = idx.unsqueeze(0).to(params.device)

        loss = torch.zeros(1, device=params.device)
        hidden = model.init_hidden(batch_size)
        cell = model.init_cell(batch_size)

        for t in range(params.train_window):
            # if z_t is missing, replace it by output mu from the last time step
            zero_index = (train_batch[t, :, 0] == 0)
            if t > 0 and torch.sum(zero_index) > 0:
                train_batch[t, zero_index, 0] = mu[zero_index]
            mu, sigma, hidden, cell = model(train_batch[t].unsqueeze_(0).clone(), idx, hidden, cell)
            loss += loss_fn(mu, sigma, labels_batch[t])

        loss.backward()
        optimizer.step()
        loss = loss.item() / params.train_window  # loss per timestep
        loss_epoch[i] = loss
        if i % 1000 == 0:
            test_metrics = evaluate(model, loss_fn, test_loader, params, epoch, sample=sampling)
            model.train()
            logger.info(f'train_loss: {loss}')
        if i == 0:
            logger.info(f'train_loss: {loss}')
    return loss_epoch


def train_and_evaluate2(model: nn.Module,
                        train_loader: DataLoader,
                        test_loader: DataLoader,
                        optimizer: optim,
                        params: utils.Params,
                        loss_fn:None,
                        restore_file: None,
                        args: None,
                        idx: None) -> None:
    '''Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the Deep AR model
        train_loader: load train data and labels
        test_loader: load test data and labels
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        params: (Params) hyperparameters
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    '''
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(params.model_dir, restore_file + '.pth.tar')
        logger.info('Restoring parameters from {}'.format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)
    logger.info('begin training and evaluation')
    best_test_ND = float('inf')
        
    # File to save first results
    out_file = os.path.join(os.path.join('experiments', args.model_name), 'train_results.csv')
    if not os.path.isfile(out_file):
        of_connection = open(out_file, 'w')
        writer = csv.writer(of_connection)
        # Write the headers to the file
        writer.writerow(['iteration', 'epoch', 'test_metric', 'train_loss'])
        of_connection.close()
    
    train_len = len(train_loader)
    ND_summary = np.zeros(params.num_epochs)
    loss_summary = np.zeros((train_len * params.num_epochs))
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=5, verbose=True, delta=0.0001, folder=params.model_dir)
    
    for epoch in range(params.num_epochs):
        logger.info('Epoch {}/{}'.format(epoch + 1, params.num_epochs))
        loss_summary[epoch * train_len:(epoch + 1) * train_len] = train(model, optimizer, loss_fn, train_loader,
                                                                        test_loader, params, args.sampling, epoch)
        test_metrics = evaluate(model, loss_fn, test_loader, params, epoch, sample=args.sampling)
        if test_metrics['rou50'] == float('nan'):
            test_metrics['rou50'] = 100
        elif test_metrics['rou50'] == 'nan':
            test_metrics['rou50'] = 100
        elif test_metrics['rou50'] == np.nan:
            test_metrics['rou50'] = 100
        
        ND_summary[epoch] = test_metrics['rou50']
        is_best = ND_summary[epoch] <= best_test_ND

        # Save weights
        utils.save_checkpoint({'epoch': 0, #epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                               epoch=0, # to prevent extra model savings
                               is_best=is_best,
                               checkpoint=params.model_dir)

        if is_best:
            logger.info('- Found new best ND')
            best_test_ND = ND_summary[epoch]
            best_json_path = os.path.join(params.model_dir, 'metrics_test_best_weights.json')
            utils.save_dict_to_json(test_metrics, best_json_path)

        logger.info('Current Best loss is: %.5f' % best_test_ND)

        #if args.plot_figure:
        #    utils.plot_all_epoch(ND_summary[:epoch + 1], args.dataset + '_ND', params.plot_dir)
        #    utils.plot_all_epoch(loss_summary[:(epoch + 1) * train_len], args.dataset + '_loss', params.plot_dir)

        last_json_path = os.path.join(params.model_dir, 'metrics_test_last_weights.json')
        utils.save_dict_to_json(test_metrics, last_json_path)
        # Write to the csv file ('a' means append)
        of_connection = open(out_file, 'a')
        writer = csv.writer(of_connection)
        writer.writerow([idx, epoch + 1, test_metrics, loss_summary[-1]]) #loss_summary[0]??
        of_connection.close()
        logger.info('Loss_summary: ' % loss_summary[epoch * train_len:(epoch + 1) * train_len])
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        logger.info('test_metrics[rou50]: %.5f ' % test_metrics['rou50'])
        early_stopping(test_metrics['rou50'], model)
        
        if early_stopping.early_stop:
            logger.info('Early stopping')
            break
            
    with open(best_json_path) as json_file:
        best_metrics = json.load(json_file)
    return best_metrics, test_metrics


def optimize_model(hyperparameters, device="cuda"):
    """Build a LARNN and train it on given dataset."""
    # Keep track of evals
    global ITERATION
    ITERATION += 1
    
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        
    #params = {k: search_params[k][search_range[idx]] for idx, k in enumerate(sorted(search_params.keys()))}
    model_param_list = '-'.join('_'.join((k, f'{v:.2f}')) for k, v in hyperparameters.items())
    model_param = copy(param_template)
    for k, v in hyperparameters.items():
        setattr(model_param, k, v)
    # Create a new folder in parent_dir with unique_name 'job_name'
    #model_name = os.path.join('experiments', args.model_name) #->model_dir
    model_name = os.path.join(os.path.join('experiments', args.model_name), 'iteration_' + str(ITERATION)) 
    if not os.path.exists(model_name):
        os.makedirs(model_name)

    # Write parameters in json file
    json_path = os.path.join(model_name, 'params.json')
    model_param.save(json_path)
    logger.info(f'Params saved to: {json_path}')
        
    # Load the parameters from json file
    model_dir = model_name #os.path.join('experiments', model_name)
    json_path = os.path.join(model_dir, 'params.json')
    data_dir = os.path.join('experiments', args.model_name) # os.path.join(args.data_dir, args.dataset)
    assert os.path.isfile(json_path), f'No json configuration file found at {json_path}'
    params = utils.Params(json_path)

    params.relative_metrics = args.relative_metrics
    params.sampling =  args.sampling
    params.save_file = args.save_file
    params.model_dir = model_dir
    params.plot_dir = os.path.join(model_dir, 'figures')
    params.lstm_hidden_dim = int(params.lstm_hidden_dim)
    params.batch_size = int(params.batch_size)
    
    if args.plot_figure:
       # create missing directories
       try:
           os.mkdir(params.plot_dir)
       except FileExistsError:
           pass

    # use GPU if available
    cuda_exist = torch.cuda.is_available()
    # Set random seeds for reproducible experiments if necessary
    if cuda_exist:
        params.device = torch.device('cuda')
        # torch.cuda.manual_seed(240)
        logger.info('Using Cuda...')
        model = net.Net(params).cuda()
    else:
        params.device = torch.device('cpu')
        # torch.manual_seed(230)
        logger.info('Not using cuda...')
        model = net.Net(params)
    
    logger.info('Dropout, Batch size, Learning rate, Hidden units: {}, {}, {}, {}'.format(params.lstm_dropout, 
                                                                                   int(params.batch_size),
                                                                                   params.learning_rate, 
                                                                                   int(params.lstm_hidden_dim)))
    
    logger.info('Loading the datasets...')
    train_set = TrainDataset(data_dir, args.dataset, params.num_class)
    test_set = TestDataset(data_dir, args.dataset, params.num_class)
    sampler = WeightedSampler(data_dir, args.dataset) # Use weighted sampler instead of random sampler
    train_loader = DataLoader(train_set, batch_size=params.batch_size, sampler=sampler, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=params.predict_batch, sampler=RandomSampler(test_set), num_workers=1)
    logger.info('Loading complete.')

    logger.info(f'Model: \n{str(model)}')
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # fetch loss function
    loss_fn = net.loss_fn

    # Train the model
    logger.info('Starting training for {} epoch(s)'.format(params.num_epochs))
    logger.info('Args {} '.format(args))
    
    st_time = datetime.now()
    best_metrics, last_metrics = train_and_evaluate2(model,
                                                     train_loader,
                                                     test_loader,
                                                     optimizer,
                                                     params,
                                                     loss_fn,
                                                     args.restore_file,
                                                     args,
                                                     ITERATION)
    eval_time = str(datetime.now() - st_time)

    try:       
        result = {
            # Note: 'loss' in Hyperopt means 'score', so we use something else it's not the real loss.
            'loss': best_metrics['rou50'],
            'best_metrics':best_metrics,
            'last_metrics':last_metrics,
            'eval_time': eval_time,
            'space': hyperparameters,
            'iteration': ITERATION,
            'status': STATUS_OK
        }

        return result

    except Exception as err:
        err_str = str(err)
        print(err_str)
        traceback_str = str(traceback.format_exc())
        print(traceback_str)
        return {
            'status': STATUS_FAIL,
            'err': err_str,
            'traceback': traceback_str
        }

    print("\n\n")

    
def get_optimizer(device):
    # Returns a callable for Hyperopt Optimization (for `fmin`):
    return lambda search_params: (
        optimize_model(search_params, device=device)
    )


def run_a_trial():
    """Run one TPE meta optimisation step and save its results."""
    global ITERATION, results_pickle_file, search_params, args
    
    logger.info("Attempt to resume a past training if it exists:")
    try:
        # https://github.com/hyperopt/hyperopt/issues/267
        trials = pickle.load(open(results_pickle_file, "rb"))
        logger.info("Found saved Trials! Loading...")
        ITERATION = len(trials.trials)
        max_evals = args.evals + len(trials.trials)
        logger.info("Rerunning from {} trials to add another one.".format(len(trials.trials)))
    except:
        trials = Trials()
        ITERATION = 0
        max_evals = args.evals
        logger.info("Starting from scratch: new trials.")
        
    
    best = fmin(
        get_optimizer('cuda'),
        search_params,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_evals,
        rstate=np.random.RandomState(42),
        show_progressbar=False,
    )
    pickle.dump(trials, open(results_pickle_file, "wb"))
    logger.info("Optimization step is completed.")

    
def main():
    # Load the 'reference' parameters from parent_dir json file
    global param_template, gpu_ids, args, search_params, model_dir, results_pickle_file
    
    args = parser.parse_args()
    model_dir = os.path.join('experiments', args.model_name)
    json_file = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(json_file), f'No json configuration file found at {json_file}'
    param_template = utils.Params(json_file)
    
    utils.set_logger(os.path.join(model_dir, 'search.log'))
    logger.info(f'Param template: {param_template}')
    
    results_pickle_file = os.path.join(model_dir, "hyper_parameter_search.pkl")

    gpu_ids = args.gpu_ids
    logger.info(f'Running on GPU: {gpu_ids}')

    # Perform hypersearch over parameters listed below
    search_params = {
        'lstm_dropout': hp.quniform('lstm_dropout', 0.1, 0.5, 0.1),
        'batch_size': 2**hp.quniform('batch_size', 6, 10, 1),
        'lstm_hidden_dim': hp.choice('lstm_hidden_dim', [2**hp.quniform('lstm_hidden_dim_a', 5, 8, 1), 
                                                         25*(2**hp.quniform('lstm_hidden_dim_b', 0, 3, 1))]),
        'learning_rate': hp.choice('learning_rate', [5*10**-hp.quniform('learning_rate_a', 3, 4, 1),
                                                     1*10**-hp.quniform('learning_rate_b', 2, 4, 1)]),
    }
    
    run_a_trial()
    

if __name__ == '__main__':
    main()
