import os, sys, math, random, time
from datetime import datetime
import _pickle as pickle

import numpy as np
import pandas as pd

import mxnet as mx
from mxnet import autograd, gluon, nd, gpu
from mxnet.gluon import nn

from nnModels import TCN
from nnTrainer import nn_trainer

from pytorchtools import EarlyStopping
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt import base
import hyperopt
base.have_bson = False
import csv
import argparse

parser = argparse.ArgumentParser(description='TCN Model')
#parser.add_argument('--dataset', type=str, default="feature_prepare_new.pkl", help='Name of the dataset')
parser.add_argument('--data-folder', default='data', help='Parent dir of the dataset')
parser.add_argument('--pickle-name', type=str, default="electricity.pkl", help='Name of the dataset')
parser.add_argument('--model-name', type=str, default="model.pkl", help='Name of the model')
parser.add_argument('--dim', type=int, default=321, help='Dimention of input data. Default=321')
parser.add_argument('--horizon', type=int, default=24, help='Dimention of output data. Default=24')
parser.add_argument('--patience', type=int, default=25, help='Dimention of output data. Default=24')
parser.add_argument('--gpu', action='store_true', help='Use GPU.')
parser.add_argument('--epochs', type=int, default=100, help='Max epochs. Default=100')
# hpo: 
parser.add_argument('--save-folder', type=str, default="./save", help='Save dir of the optimization')
parser.add_argument('--save-file', type=str, default="train_results.csv", help='File name of the optimization results')
parser.add_argument('--hop-file', type=str, default="hyper_parameter_search.pkl", help='File name of the optimization results')
parser.add_argument('--evals', type=int, default=100, help='Max evaluations. Default=100')



def optimize(optimizer_params):
    """
    The model optimization
    """
    # Keep track of evals
    global ITERATION, out_file, epochs
    ITERATION += 1
    ### The model parameters
    abs_loss = gluon.loss.L1Loss()
    L2_loss = gluon.loss.L2Loss()
    huber_loss = gluon.loss.HuberLoss()
    initializer = mx.initializer.MSRAPrelu()
    optimizer = 'adam';
    optimizer_params['units'] = int(optimizer_params['units'])
    optimizer_params['batch_size'] = int(optimizer_params['batch_size'])
    print("Iteration %d: Creating model" % ITERATION)
    
    model1 = TCN(input_dimention=args.dim, 
                 output_ax=args.horizon, 
                 units=optimizer_params['units'], 
                 dropout=optimizer_params['dropout'])
        
    train_mark='optimizing'
    
    #epochs = int(batch/batches[0])
    trainer_params_list = {'batch_size': optimizer_params['batch_size'],
                           'epoch_num': epochs,
                           'loss_func': huber_loss, 
                           'initializer': initializer,
                           'optimizer':optimizer, 
                           'optimizer_params':{'learning_rate':optimizer_params['learning_rate']},
                           'patience': args.patience,
                           'iteration':ITERATION,
                           'units':optimizer_params['units'],
                           'dropout':optimizer_params['dropout'],
                           'lr':optimizer_params['learning_rate']}
    
    print("Training model ... ")
    st_time = datetime.now() 
    
    valid_loss, valid_ND, valid_NRMSE, rho10, rho50, rho90 = nn_trainer(train_mark, model1, 
                                                                       test_sub_valid_X, test_future_test_X, test_sub_valid_Y,
                                                                       sub_train_nd, sub_valid_X,future_test_X, sub_valid_Y, 
                                                                       trainer_params_list=trainer_params_list, 
                                                                       ctx=ctx,
                                                                       model_name=args.model_name)
        
    eval_time = str(datetime.now() - st_time)
    print("Iteration %d: Getting results ... " % ITERATION)
    
    return {'loss': valid_ND, 
            'ND':valid_ND, 
            'NRMSE':valid_NRMSE, 
            'val_loss':valid_loss,
            'params': optimizer_params,
            'rho_metric':{'rho10': rho10, 'rho50':rho50, 'rho90':rho90},
            'iteration': ITERATION,
            'eval_time':eval_time,
            'status': STATUS_OK}


if __name__ == '__main__':
    
    global out_file, epochs

    args = parser.parse_args()
    
    ### The input dataset
    if args.gpu:
        ctx = mx.gpu(0)
    else:
        ctx = mx.cpu(0)
    
    ### save file 
    out_file = os.path.join('./save', 'train_results_' + args.model_name[:-4] + '.csv')
    if not os.path.isfile(out_file):
        of_connection = open(out_file, 'w')
        writer = csv.writer(of_connection)
        # Write the headers to the file
        writer.writerow(['params', 'epoch', 'train_loss', 'valid_loss', 'val_ND', 'val_NRMSE', 'val_rho10', 'val_rho50', 'val_rho90', 'time'])
        of_connection.close()
        
    ### train and validation data
    
    #file_name = args.dataset #'feature_prepare.pkl'
    with open(os.path.join(args.data_folder, args.pickle_name), 'rb') as f:
        [trainX_dt,trainX2_dt, trainY_dt,trainY2_dt, testX_dt, testX2_dt,testY_dt,testY2_dt] = pickle.load(f)

    sub_train_X, sub_train_Y = nd.array(trainX_dt, ctx=ctx), nd.array(trainY_dt, ctx=ctx)
    sub_valid_X, sub_valid_Y = nd.array(testX_dt, ctx=ctx), nd.array(testY_dt, ctx=ctx)
    future_train_X, future_test_X = nd.array(trainY2_dt, ctx=ctx), nd.array(testY2_dt, ctx=ctx)

    sub_train_nd = gluon.data.ArrayDataset(sub_train_X, future_train_X, sub_train_Y)
    
    # just a copy of validation set
    test_sub_valid_X = sub_valid_X
    test_sub_valid_Y = sub_valid_Y
    test_future_test_X = future_test_X
    
    # File to save first results

    save_dir = args.save_folder # './save'
    save_file = args.save_file #'search_results.csv'
    hop_file = args.hop_file #"hyper_parameter_search.pkl"
    nb_evals = args.evals
    epochs = args.epochs
    out_file = os.path.join(save_dir, save_file)
    results_pickle_file = os.path.join(save_dir, hop_file)

    try:
        os.mkdir(save_dir)
    except FileExistsError:
        pass

#     if not os.path.isfile(out_file):
#         of_connection = open(out_file, 'w')
#         writer = csv.writer(of_connection)
#         # Write the headers to the file
#         writer.writerow(['iteration', 'batch_size', 'epoch', 'hyper-parameters', 'val_metric', 'val_metric2', 'time'])
#         of_connection.close()
    
    # Global variable
    global  ITERATION
    
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
                     'batch_size': 2**hp.quniform('batch_size', 6, 10, 1),
                     'units': hp.choice('units', [2**hp.quniform('units_a', 5, 8, 1), 
                                                  25*(2**hp.quniform('units_b', 0, 3, 1))]),
                     'learning_rate': hp.choice('learning_rate', [5*10**-hp.quniform('lr_a', 3, 4, 1),
                                                                  1*10**-hp.quniform('lr_b', 2, 4, 1)])}

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
    
    
    
    
    