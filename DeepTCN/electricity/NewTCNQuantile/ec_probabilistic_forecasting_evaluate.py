import os, sys, math, random, time
from datetime import datetime
import _pickle as pickle

import numpy as np
import pandas as pd

import mxnet as mx
from mxnet import autograd, gluon, nd, gpu
from mxnet.gluon import nn

from nnModels import TCN
from nnTrainer import nn_trainer, DLPred2
from nnHelper import smape,rmsle,ND,NRMSE,rho_risk,rho_risk2,group_ND,group_NRMSE,group_rho_risk
from nnModels import QuantileLoss

from pytorchtools import EarlyStopping
import csv
import argparse

parser = argparse.ArgumentParser(description='TCN Model')
parser.add_argument('--data-folder', default='data', help='Parent dir of the dataset')
parser.add_argument('--pickle-name', type=str, default="electricity.pkl", help='Name of the dataset')
parser.add_argument('--model-name', type=str, default="model.pkl", help='Name of the model')
parser.add_argument('--dim', type=int, default=321, help='Dimention of input data. Default=321')
parser.add_argument('--horizon', type=int, default=24, help='Dimention of output data. Default=24')
parser.add_argument('--gpu', action='store_true', help='whenever to use GPU.')
# parser.add_argument('--epochs', type=int, default=500, help='Max epochs. Default=100')
# parser.add_argument('--batch_size', type=int, default=512, help='Max epochs. Default=100')
# parser.add_argument('--units', type=int, default=64, help='Dense layer units. Default=64')
# parser.add_argument('--learning_rate', type=float, default=0.5, help='Optimizer learning rate. Default=0.5')
# parser.add_argument('--dropout', type=float, default=0.2, help='Dropout for the dense layer. Default=0.2')
# parser.add_argument('--patience', type=int, default=25, help='Early Stopping patience. Default=25')


if __name__ == '__main__':
    
    args = parser.parse_args()

    ### The input dataset
    if args.gpu:
        ctx = mx.gpu(0)
    else:
        ctx = mx.cpu(0)
    
    with open(os.path.join(args.data_folder, args.pickle_name), 'rb') as f:
        [trainX_dt,trainX2_dt, trainY_dt,trainY2_dt, testX_dt, testX2_dt,testY_dt,testY2_dt] = pickle.load(f)

    sub_train_X, sub_train_Y = nd.array(trainX_dt, ctx=ctx), nd.array(trainY_dt, ctx=ctx)
    sub_valid_X, sub_valid_Y = nd.array(testX_dt, ctx=ctx), nd.array(testY_dt, ctx=ctx)
    future_train_X, future_test_X = nd.array(trainY2_dt, ctx=ctx), nd.array(testY2_dt, ctx=ctx)
    sub_train_nd = gluon.data.ArrayDataset(sub_train_X, future_train_X, sub_train_Y)
    
    
    ### load model
    
    model1 = pickle.load(open(os.path.join('./save', args.model_name), 'rb'))
    loss_func = gluon.loss.L1Loss()

    """
    The model evaluation
    """
    ### The model parameters  val_test_conv_X, val_test_data_X, val_test_data_Y,
    
    valid_true = sub_valid_Y.asnumpy()
    valid_predQ10, valid_predQ50, valid_predQ90 = DLPred2(model1, sub_valid_X, future_test_X)
    valid_predQ10 = valid_predQ10.asnumpy()
    valid_predQ50 = valid_predQ50.asnumpy()
    valid_predQ90 = valid_predQ90.asnumpy()

    valid_pred = valid_predQ50.copy()
    #valid_true = val_test_data_Y.asnumpy()
    ### The evaluation
    valid_loss = nd.sum(loss_func(nd.array(valid_true), nd.array(valid_predQ50))).asscalar()
    rho10 = rho_risk2(valid_predQ10.reshape(-1,), valid_true.reshape(-1,), 0.1)
    rho50 = rho_risk2(valid_predQ50.reshape(-1,), valid_true.reshape(-1,), 0.5)
    rho90 = rho_risk2(valid_predQ90.reshape(-1,), valid_true.reshape(-1,), 0.9)
    #print(valid_true.shape)
    #print(valid_true[0,:])
    #print(valid_pred[0,:])
    #valid_pred = valid_pred.reshape(-1,)
    #valid_true = valid_true.reshape(-1,)
    valid_pred2 = valid_pred[valid_true>0]
    valid_true2 = valid_true[valid_true>0]
    valid_loss = nd.sum(loss_func(nd.array(valid_true), nd.array(valid_pred))).asscalar()
    valid_ND = ND(valid_pred, valid_true);  valid_ND2 = ND(valid_pred2, valid_true2)
    valid_NRMSE = NRMSE(valid_pred, valid_true); valid_NRMSE2 = NRMSE(valid_pred2, valid_true2)
    print("FINAL valid loss: %f valid ND: %f, valid NRMSE %f" % (valid_loss, valid_ND,valid_NRMSE))
    print("FINAL valid loss: %f valid ND: %f, valid NRMSE %f" % (valid_loss, valid_ND2,valid_NRMSE2))
    print("FINAL valid loss: %f valid rho-risk 10: %f, valid rho-risk 50: %f,  valid rho-risk 90: %f" % (valid_loss, rho10, rho50, rho90))
    
    ## save predictions
    np.save('./save/' + args.model_name[:-4] + '_q_10.npy', valid_predQ10)
    np.save('./save/' + args.model_name[:-4] + '_q_50.npy', valid_predQ50)
    np.save('./save/' + args.model_name[:-4] + '_q_90.npy', valid_predQ90)
    np.save('./save/' + args.model_name[:-4] + '_true.npy', valid_true)

