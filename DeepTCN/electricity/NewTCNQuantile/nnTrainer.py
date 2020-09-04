import sys, os, math, random, time, datetime
import _pickle as pickle

import numpy as np
import pandas as pd

import mxnet as mx
from mxnet import autograd, gluon, nd, gpu
from mxnet.gluon import nn,rnn

from tqdm import trange

from nnHelper import smape,rmsle,ND,NRMSE,rho_risk,rho_risk2,group_ND,group_NRMSE,group_rho_risk
from nnModels import QuantileLoss

from pytorchtools import EarlyStopping
import csv

### check point: save the temporal model params
def save_checkpoint(net, mark, valid_metric, save_path):
    if not path.exists(save_path):
        os.makedirs(save_path)
    filename = path.join(save_path, "mark_{:s}_metrics_{:.3f}".format(mark, valid_metric))
    filename +='.param'
    net.save_params(filename)

    
def DLPred(net, dt):
    if(dt.shape[0]<=60000):
        print(type(net(conv_dt, dt)))
        return net(dt)
    block_size = dt.shape[0] //60000+1
    pred_result = net(dt[0:60000,])
    for i in range(1,block_size):
        i = i*60000
        j = min(i+60000, dt.shape[0])
        block_pred = net(dt[i:j, ])
        pred_result = nd.concat(pred_result, block_pred, dim=0)
    return pred_result


"""
The main training process
"""
def DLPred2(net,conv_dt ,dt):
    if(dt.shape[0]<=60000):
        return net(conv_dt, dt)
    block_size = dt.shape[0] //60000+1
    pred_result = net(conv_dt[0:60000,], dt[0:60000,])
    pred_result1 = pred_result[0]
    pred_result2 = pred_result[1]
    pred_result3 = pred_result[2]    
    for i in range(1,block_size):
        i = i*60000
        j = min(i+60000, dt.shape[0])
        block_pred = net(conv_dt[i:j, ], dt[i:j, ])
        #print('pred_result: ', pred_result.shape)
        #print('block_pred: ', block_pred.shape) 
        pred_result1 = nd.concat(pred_result1, block_pred[0], dim=0)
        pred_result2 = nd.concat(pred_result2, block_pred[1], dim=0)
        pred_result3 = nd.concat(pred_result3, block_pred[2], dim=0)
        print(pred_result3.shape)
    #print('sss')
    return pred_result1, pred_result2, pred_result3 


"""
The main training process
"""
def nn_trainer(train_mark, model, val_test_conv_X, val_test_data_X, val_test_data_Y, train_data, test_conv_X, test_data_X, test_data_Y, trainer_params_list, ctx, model_name):
    """Parsing the params list"""
    ### The data
    batch_size = trainer_params_list['batch_size']
    epoches = trainer_params_list['epoch_num']

    loss_func = trainer_params_list['loss_func']
    initializer = trainer_params_list['initializer']
    optimizer = trainer_params_list['optimizer']
    optimizer_params = trainer_params_list['optimizer_params']
    patience = trainer_params_list['patience']
     ### The quantile loss
    loss10 = QuantileLoss(quantile_alpha=0.1)
    loss50= QuantileLoss(quantile_alpha=0.5)
    loss90 = QuantileLoss(quantile_alpha=0.9)

    #train_iter = gluon.data.DataLoader(train_data, batch_size, shuffle=True)
    ### The model
    mx.random.seed(123456)
    model.collect_params().initialize(initializer, ctx=ctx)
    trainer = gluon.Trainer(model.collect_params(),optimizer=optimizer, optimizer_params=optimizer_params)
    n_train = len(train_data)
    n_test = len(test_data_Y)
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True, delta=0.0001, model_name=model_name)
    
    ### The quantile loss
    ### The training process
    for epoch in trange(epoches):
        start=time.time()
        start_time = datetime.datetime.now()
        train_loss = 0
        k = 0
        train_iter = gluon.data.DataLoader(train_data, batch_size, shuffle=True)
        for conv_data, data, label in train_iter:
            label = label.as_in_context(ctx)
            with autograd.record():
                outputQ10, outputQ50, outputQ90 = model(conv_data, data)
                lossQ10 = loss10(outputQ10, label)
                lossQ50 = loss50(outputQ50, label)
                lossQ90 = loss90(outputQ90, label)
                loss = (lossQ10+lossQ50+lossQ90)#*weight
            loss.backward()
            trainer.step(batch_size=1,ignore_stale_grad=True)
            train_loss += nd.sum(loss).asscalar()
            k += 1
            if k*batch_size>300000: 
                print('training_data_nb:',k*batch_size)
                break
        ### The test loss
        ## The valid_true
        valid_true = test_data_Y.asnumpy()
        valid_predQ10, valid_predQ50, valid_predQ90 = DLPred2(model, test_conv_X, test_data_X)
        valid_predQ10 = valid_predQ10.asnumpy()
        valid_predQ50 = valid_predQ50.asnumpy()
        valid_predQ90 = valid_predQ90.asnumpy()
        valid_pred = valid_predQ50.copy()
        valid_true = test_data_Y.asnumpy()
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
        print("Epoch %d, valid loss: %f valid ND: %f, valid NRMSE %f" % (epoch, valid_loss, valid_ND,valid_NRMSE))
        print("Epoch %d, valid loss: %f valid ND: %f, valid NRMSE %f" % (epoch, valid_loss, valid_ND2,valid_NRMSE2))
        print("Epoch %d, valid loss: %f valid rho-risk 10: %f, valid rho-risk 50: %f,  valid rho-risk 90: %f" % (epoch, valid_loss, rho10, rho50, rho90))
        
        end_time = str(datetime.datetime.now() - start_time)
        # Write to the csv file ('a' means append)
        of_connection = open(os.path.join('./save', 'train_results_' + model_name[:-4]) + '.csv', 'a')
        writer = csv.writer(of_connection)
        writer.writerow([trainer_params_list, epoch, train_loss, valid_loss, valid_ND, valid_NRMSE, rho10, rho50, rho90, end_time])
        of_connection.close()

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_ND, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    if train_mark=='testing':
        # The valid_true
        print('---')
        print('Validation agains test data:')
        print('---')
        ## load the best model
        best_model = pickle.load(open(os.path.join('./save', model_name), 'rb'))
        ##

        models = [model, best_model]
        for	net in models:
            valid_true = val_test_data_Y.asnumpy()
            valid_predQ10, valid_predQ50, valid_predQ90 = DLPred2(net, val_test_conv_X, val_test_data_X)
            valid_predQ10 = valid_predQ10.asnumpy()
            valid_predQ50 = valid_predQ50.asnumpy()
            valid_predQ90 = valid_predQ90.asnumpy()

            valid_pred = valid_predQ50.copy()
            valid_true = val_test_data_Y.asnumpy()
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
            print("FINAL Epoch %d, valid loss: %f valid ND: %f, valid NRMSE %f" % (epoch, valid_loss, valid_ND,valid_NRMSE))
            print("FINAL Epoch %d, valid loss: %f valid ND: %f, valid NRMSE %f" % (epoch, valid_loss, valid_ND2,valid_NRMSE2))
            print("FINAL Epoch %d, valid loss: %f valid rho-risk 10: %f, valid rho-risk 50: %f,  valid rho-risk 90: %f" % (epoch, valid_loss, rho10, rho50, rho90))

        ## save predictions
        np.save('./save/' + model_name[:-4] + '_q_10.npy', valid_predQ10)
        np.save('./save/' + model_name[:-4] + '_q_50.npy', valid_predQ50)
        np.save('./save/' + model_name[:-4] + '_q_90.npy', valid_predQ90)
        np.save('./save/' + model_name[:-4] + '_true.npy', valid_true)
    
    else:
            # The valid_true
        print('---')
        print('Validation agains test data:')
        print('---')
        ## load the best model
        best_model = pickle.load(open(os.path.join('./save', model_name), 'rb'))
        ##
        #for	net in models:
        valid_true = val_test_data_Y.asnumpy()
        valid_predQ10, valid_predQ50, valid_predQ90 = DLPred2(best_model, val_test_conv_X, val_test_data_X)
        valid_predQ10 = valid_predQ10.asnumpy()
        valid_predQ50 = valid_predQ50.asnumpy()
        valid_predQ90 = valid_predQ90.asnumpy()

        valid_pred = valid_predQ50.copy()
        valid_true = val_test_data_Y.asnumpy()
        ### The evaluation
        valid_loss = nd.sum(loss_func(nd.array(valid_true), nd.array(valid_predQ50))).asscalar()
        rho10 = rho_risk2(valid_predQ10.reshape(-1,), valid_true.reshape(-1,), 0.1)
        rho50 = rho_risk2(valid_predQ50.reshape(-1,), valid_true.reshape(-1,), 0.5)
        rho90 = rho_risk2(valid_predQ90.reshape(-1,), valid_true.reshape(-1,), 0.9)
        
        valid_pred2 = valid_pred[valid_true>0]
        valid_true2 = valid_true[valid_true>0]
        
        valid_loss = nd.sum(loss_func(nd.array(valid_true), nd.array(valid_pred))).asscalar()
        valid_ND = ND(valid_pred, valid_true);  valid_ND2 = ND(valid_pred2, valid_true2)
        valid_NRMSE = NRMSE(valid_pred, valid_true); valid_NRMSE2 = NRMSE(valid_pred2, valid_true2)
        #print("FINAL Epoch %d, valid loss: %f valid ND: %f, valid NRMSE %f" % (epoch, valid_loss, valid_ND,valid_NRMSE))
        #print("FINAL Epoch %d, valid loss: %f valid ND: %f, valid NRMSE %f" % (epoch, valid_loss, valid_ND2,valid_NRMSE2))
        #print("FINAL Epoch %d, valid loss: %f valid rho-risk 10: %f, valid rho-risk 50: %f,  valid rho-risk 90: %f" % (epoch, valid_loss, rho10, rho50, rho90))
        return valid_loss, valid_ND, valid_NRMSE, rho10, rho50, rho90

