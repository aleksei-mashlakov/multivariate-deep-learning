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
import csv
import argparse

parser = argparse.ArgumentParser(description='TCN Model')
parser.add_argument('--data-folder', default='data', help='Parent dir of the dataset')
parser.add_argument('--pickle-name', type=str, default="electricity.pkl", help='Name of the dataset')
parser.add_argument('--model-name', type=str, default="model.pkl", help='Name of the model')
parser.add_argument('--dim', type=int, default=321, help='Dimention of input data. Default=321')
parser.add_argument('--horizon', type=int, default=24, help='Dimention of output data. Default=24')
parser.add_argument('--patience', type=int, default=25, help='Early Stopping patience. Default=25')
parser.add_argument('--gpu', action='store_true', help='whenever to use GPU.')
parser.add_argument('--epochs', type=int, default=500, help='Max epochs. Default=100')
parser.add_argument('--batch_size', type=int, default=512, help='Max epochs. Default=100')
parser.add_argument('--units', type=int, default=64, help='Dense layer units. Default=64')
parser.add_argument('--learning_rate', type=float, default=0.5, help='Optimizer learning rate. Default=0.5')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout for the dense layer. Default=0.2')


if __name__ == '__main__':
    
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
    
    with open(os.path.join(args.data_folder, args.pickle_name), 'rb') as f:
        [trainX_dt,trainX2_dt, trainY_dt,trainY2_dt, testX_dt, testX2_dt,testY_dt,testY2_dt] = pickle.load(f)
        
    ### split testing data into validation and test
    
    split = int(0.5*testY_dt.shape[0])
    
    val_testX_dt = testX_dt[split:, :]
    val_testY_dt = testY_dt[split:, :]
    val_testY2_dt = testY2_dt[split:, :, :]
    
    testX_dt = testX_dt[:split, :]
    testY_dt = testY_dt[:split, :]
    testY2_dt = testY2_dt[:split, :, :]

    sub_train_X, sub_train_Y = nd.array(trainX_dt, ctx=ctx), nd.array(trainY_dt, ctx=ctx)
    sub_valid_X, sub_valid_Y = nd.array(testX_dt, ctx=ctx), nd.array(testY_dt, ctx=ctx)
    future_train_X, future_test_X = nd.array(trainY2_dt, ctx=ctx), nd.array(testY2_dt, ctx=ctx)

    sub_train_nd = gluon.data.ArrayDataset(sub_train_X, future_train_X, sub_train_Y)
    
    test_sub_valid_X, test_sub_valid_Y = nd.array(val_testX_dt, ctx=ctx), nd.array(val_testY_dt, ctx=ctx)
    test_future_test_X = nd.array(val_testY2_dt, ctx=ctx)
    
    ### load model
    
    model1 = TCN(input_dimention=args.dim, output_ax=args.horizon, units=args.units, dropout=args.dropout)
    #choose parameters
    
    """
    The model training
    """
    ### The model parameters
    abs_loss = gluon.loss.L1Loss()
    L2_loss = gluon.loss.L2Loss()
    huber_loss = gluon.loss.HuberLoss()
    initializer = mx.initializer.MSRAPrelu()
    optimizer = 'adam';
    optimizer_params = {'learning_rate': args.learning_rate}

    trainer_params_list = {'batch_size': args.batch_size, 
                           'epoch_num': args.epochs,
                           'loss_func': abs_loss, 
                           'initializer': initializer,
                           'optimizer': optimizer, 
                           'optimizer_params': optimizer_params,
                           'patience': args.patience}
    
    train_mark='testing'

    nn_trainer(train_mark, model1, 
               test_sub_valid_X, test_future_test_X, test_sub_valid_Y,
               sub_train_nd, 
               sub_valid_X, future_test_X, sub_valid_Y, 
               trainer_params_list=trainer_params_list, ctx=ctx, 
               model_name=args.model_name)

