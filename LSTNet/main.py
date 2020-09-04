####################################################################################
#   Implementation of the following paper: https://arxiv.org/pdf/1703.07015.pdf    #
#                                                                                  #
#    Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks     #
####################################################################################

# This must be set in the beggining because in model_util, we import it
logger_name = "lstnet"

# Path appended in order to import from util
import sys, os
sys.path.append('..')

import pickle
import json
import csv
import numpy as np
import pandas as pd
#import pyyaml
#import h5py
#!pip install -q pyyaml h5py  # Required to save models in HDF5 format


from util.model_util import LoadModel, SaveModel, SaveResults, SaveHistory
from util.Msglog import LogInit

from datetime import datetime

from lstnet_util import GetArguments, LSTNetInit
from lstnet_datautil import DataUtil
from lstnet_model import PreSkipTrans, PostSkipTrans, PreARTrans, PostARTrans, LSTNetModel, ModelCompile
from lstnet_plot import AutoCorrelationPlot, PlotHistory, PlotPrediction


from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, STATUS_FAIL
from hyperopt import base
import hyperopt
#base.have_bson = False

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import History, ModelCheckpoint, CSVLogger, EarlyStopping, TerminateOnNaN

# tf.reset_default_graph()
#tf.set_random_seed(0)
tf.random.set_seed(0)
import random
random.seed(0)
np.random.seed(0)

custom_objects = {
        'PreSkipTrans': PreSkipTrans,
        'PostSkipTrans': PostSkipTrans,
        'PreARTrans': PreARTrans,
        'PostARTrans': PostARTrans
        }


def train(model, data, init, batch, epochs, tensorboard = None):

    if init.validate == True:
        val_data = (data.valid[0], data.valid[1])
    else:
        val_data = None

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=init.patience, verbose=1, mode='auto')
    mcp_save = ModelCheckpoint(init.save + '.h5', save_best_only=True, save_weights_only=True,
                               monitor='val_loss', mode='min')

    ## train the model
    start_time = datetime.now()
    history = model.fit(
                x = data.train[0],
                y = data.train[1],
                epochs = epochs, #init.epochs,
                batch_size = batch, #init.batchsize,
                validation_data = val_data,
                callbacks = [early_stop, mcp_save]#, TerminateOnNaN(), tensorboard] if tensorboard else None
            )
    end_time = datetime.now()
    log.info("Training time with batch %d took: %s", batch, str(end_time - start_time))

    return history


def optimize_model(hyperparameters):
    global ITERATION, lstnet_init, Data
    ITERATION += 1

    lstnet_init.lr = hyperparameters['lr']
    lstnet_init.dropout = round(hyperparameters['dropout'], 1)
    lstnet_init.GRUUnits = int(hyperparameters['GRUUnits'])
    lstnet_init.batchsize = int(hyperparameters['batchsize'])

    log.info("Hyper-parameters: Learning rate:%f, Batch_size:%f, Dropout:%f, GRUUnits:%f",
             lstnet_init.lr,
             lstnet_init.batchsize,
             lstnet_init.dropout,
             lstnet_init.GRUUnits)

    log.info("Creating model")
    lstnet = LSTNetModel(lstnet_init, Data.train[0].shape)

    log.info("Compile model")
    lstnet_tensorboard = ModelCompile(lstnet, lstnet_init)

    if lstnet_tensorboard is not None:
        log.info("Model compiled ... Open tensorboard in order to visualise it!")
    else:
        log.info("Model compiled ... No tensorboard visualisation is available")

    log.info("Training model ... ")

    start_time = datetime.now()
    h = train(lstnet, Data, lstnet_init, lstnet_init.batchsize, lstnet_init.epochs, lstnet_tensorboard)
    time = datetime.now() - start_time
    loss, rse, corr, nrmse, nd = lstnet.evaluate(Data.valid[0], Data.valid[1]) #
    log.info("Validation on the validation set returned: Loss:%f, RSE:%f, Correlation:%f, NRMSE:%f, ND:%f", loss, rse, corr, nrmse, nd)
    # Write to the csv file
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    for i in range(1, len(h.history['loss'])+1):   #(1,lstnet_init.epochs+1):
        train_metric = {'train_mae': h.history['loss'][i-1],
                        'train_rse': h.history['rse'][i-1],
                        'train_corr': h.history['corr'][i-1],
                        'train_nrmse': h.history['nrmse'][i-1],
                        'train_nd': h.history['nd'][i-1]}

        test_metric = {'test_mae': h.history['val_loss'][i-1],
                       'test_rse': h.history['val_rse'][i-1],
                       'test_corr': h.history['val_corr'][i-1],
                       'test_nrmse': h.history['val_nrmse'][i-1],
                       'test_nd': h.history['val_nd'][i-1]}

        writer.writerow([ITERATION, i, train_metric, test_metric, hyperparameters, time])
    of_connection.close()

    eval_time = str(datetime.now()-start_time)

    if (nd == np.inf):
        nd = 1.
    elif (nd == np.nan):
        nd = 1.
    elif nd == 'nan':
        nd = 1. 

    result = {'loss': nd,
              'rse':rse,
              'corr':corr,
              'rse':nrmse,
              'nd':nd,
              'val_loss': loss,
              'space': hyperparameters,
              'iteration': ITERATION,
              'eval_time': eval_time,
              'status': STATUS_OK}

    return result


if __name__ == '__main__':
    try:
        args = GetArguments()
    except SystemExit as err:
        print("Error reading arguments")
        exit(0)

    test_result = None

    # Initialise parameters
    global lstnet_init, Data, results_pickle_file
    lstnet_init = LSTNetInit(args)

    # Initialise logging
    log = LogInit(logger_name, lstnet_init.logfilename, lstnet_init.debuglevel, lstnet_init.log)
    log.info("Python version: %s", sys.version)
    log.info("Tensorflow version: %s", tf.__version__)
    log.info("Keras version: %s ... Using tensorflow embedded keras", tf.keras.__version__)

    # Dumping configuration
    lstnet_init.dump()

    # Reading data
    Data = DataUtil(lstnet_init.data,
                    lstnet_init.trainpercent,
                    lstnet_init.validpercent,
                    lstnet_init.horizon,
                    lstnet_init.window,
                    lstnet_init.normalise)

    # If file does not exist, then Data will not have attribute 'data'
    if hasattr(Data, 'data') is False:
        log.critical("Could not load data!! Exiting")
        exit(1)

    log.info("Training shape: X:%s Y:%s", str(Data.train[0].shape), str(Data.train[1].shape))
    log.info("Validation shape: X:%s Y:%s", str(Data.valid[0].shape), str(Data.valid[1].shape))
    log.info("Testing shape: X:%s Y:%s", str(Data.test[0].shape), str(Data.test[1].shape))


    # Hyperparameter optimization
    if lstnet_init.optimize == True:
        log.info("Optimizing model parameters... ")

        # Perform hypersearch over parameters listed below
        search_params = {
            'dropout': hp.quniform('dropout', 0.1, 0.5, 0.1),
            'batchsize': 2**hp.quniform('batchsize', 6, 10, 1),
            'GRUUnits': hp.choice('GRUUnits', [2**hp.quniform('GRUUnits_a', 5, 8, 1),
                                               25*(2**hp.quniform('GRUUnits_b', 0, 3, 1))]),
            'lr': hp.choice('lr', [5*10**-hp.quniform('lr_a', 3, 4, 1),
                                   1*10**-hp.quniform('lr_b', 2, 4, 1)]),
        }

        """Run one TPE meta optimisation step and save its results."""

        # File to save first results
        global out_file
        out_file = lstnet_init.save + '_train_results.csv'

        if not os.path.isfile(out_file):
            of_connection = open(out_file, 'w')
            writer = csv.writer(of_connection)
            # Write the headers to the file
            writer.writerow(['iteration', 'epoch', 'train_metric', 'test_metric', 'space', 'time'])
            of_connection.close()

        log.info("Attempt to resume a past training if it exists:")

        results_pickle_file = lstnet_init.save + "_hyper_parameter_search.pkl"
        global ITERATION
        try:
            # https://github.com/hyperopt/hyperopt/issues/267
            trials = pickle.load(open(results_pickle_file, "rb"))
            log.info("Found saved Trials! Loading...")
            max_evals = lstnet_init.evals + len(trials.trials)
            ITERATION = len(trials.trials)
            log.info("Rerunning from {} trials to add another one.".format(len(trials.trials)))
        except:
            trials = Trials()
            max_evals = lstnet_init.evals
            ITERATION = 0
            log.info("Starting from scratch: new trials.")

        best = fmin(optimize_model,
                    search_params,
                    algo=tpe.suggest,
                    trials=trials,
                    max_evals=max_evals,
                    rstate=np.random.RandomState(42),
                    show_progressbar=False)

        pickle.dump(trials, open(results_pickle_file, "wb"))

        trials_results = sorted(trials.results, key = lambda x: x['loss'])
        best_params = trials_results[0]['space']
        log.info("Optimization step complete.")
        log.info("Optimization parameters {}" % best_params)
        lstnet_init.lr = best_params['lr']
        lstnet_init.dropout = round(best_params['dropout'], 1)
        lstnet_init.GRUUnits = int(best_params['GRUUnits'])
        lstnet_init.batchsize = int(best_params['batchsize'])

        # limit the run to optimization task only
        lstnet_init.train = False
        lstnet_init.validation = False
        lstnet_init.evaltest = False
        lstnet_init.predict = None


    if lstnet_init.plot == True and lstnet_init.autocorrelation is not None:
        AutoCorrelationPlot(Data, lstnet_init)

    # If --load is set, load model from file, otherwise create model
    if lstnet_init.load is not None:
        log.info("Load model from %s", lstnet_init.load)
        lstnet = LoadModel(lstnet_init.load, custom_objects)
        ######################################################################################               added
        #lstnet = LSTNetModel(lstnet_init, Data.train[0].shape)
        #lstnet.load_weights(os.path.join(lstnet_init.load, 'best_model.h5'))

    else:
        log.info("Creating model")
        lstnet = LSTNetModel(lstnet_init, Data.train[0].shape)

    if lstnet is None:
        log.critical("Model could not be loaded or created ... exiting!!")
        exit(1)

    # Compile model
    lstnet_tensorboard = ModelCompile(lstnet, lstnet_init)
    if lstnet_tensorboard is not None:
        log.info("Model compiled ... Open tensorboard in order to visualise it!")
    else:
        log.info("Model compiled ... No tensorboard visualisation is available")

    # Model Training
    if lstnet_init.train is True:
        # Train the model
        log.info("Training model ... ")
        h = train(lstnet, Data, lstnet_init, lstnet_init.batchsize, lstnet_init.epochs,  lstnet_tensorboard)

        # Plot training metrics
        if lstnet_init.plot is True:
            PlotHistory(h.history, ['loss', 'rse', 'corr', 'nrmse', 'nd'], lstnet_init)

        # Saving model if lstnet_init.save is not None.
        # There's no reason to save a model if lstnet_init.train == False
        SaveModel(lstnet, lstnet_init.save)
        if lstnet_init.saveresults == True:
            SaveResults(lstnet, lstnet_init, h.history, test_result, ['loss', 'rse', 'corr', 'nrmse', 'nd'])
        if lstnet_init.savehistory == True:
            SaveHistory(lstnet_init.save, h.history)

    # Validation
    if lstnet_init.train is False and lstnet_init.validate is True:
        loss, rse, corr, nrmse, nd = lstnet.evaluate(Data.valid[0], Data.valid[1])
        log.info("Validation on the validation set returned: Loss:%f, RSE:%f, Correlation:%f, NRMSE:%f, ND:%f",
                 loss, rse, corr, nrmse, nd)
    elif lstnet_init.validate == True:
        log.info("Validation on the validation set returned: Loss:%f, RSE:%f, Correlation:%f, NRMSE:%f, ND:%f",
                 h.history['val_loss'][-1], h.history['val_rse'][-1], h.history['val_corr'][-1], h.history['val_nrmse'][-1], h.history['val_nd'][-1])

    # Testing evaluation
    if lstnet_init.evaltest is True:
        loss, rse, corr, nrmse, nd = lstnet.evaluate(Data.test[0], Data.test[1])
        log.info("Validation on the test set returned: Loss:%f, RSE:%f, Correlation:%f, NRMSE:%f, ND:%f",
                 loss, rse, corr, nrmse, nd)
        test_result = {'loss': loss, 'rse': rse, 'corr': corr, 'nrmse': nrmse, 'nd': nd}

    # Prediction
    if lstnet_init.predict is not None:
        if lstnet_init.predict == 'trainingdata' or lstnet_init.predict == 'all':
            log.info("Predict training data")
            trainPredict = lstnet.predict(Data.train[0])
        else:
            trainPredict = None
        if lstnet_init.predict == 'validationdata' or lstnet_init.predict == 'all':
            log.info("Predict validation data")
            validPredict = lstnet.predict(Data.valid[0])
        else:
            validPredict = None
        if lstnet_init.predict == 'testingdata' or lstnet_init.predict == 'all':
            log.info("Predict testing data")
            start_time = datetime.now()
            testPredict = lstnet.predict(Data.test[0], verbose=1)
            log.info("Validation on one test set took: %s", str(datetime.now()-start_time))
            start_time = datetime.now()
            Yt_hat = np.array([lstnet.predict(Data.test[0], verbose=0) for _ in range(lstnet_init.mc_iterations)])
            timedelta = datetime.now()-start_time
            log.info("Validation on the test set took: %s", str(datetime.now()-start_time))
            log.info("Validation on the test set took: %s per prediction", str(timedelta.total_seconds()/(Data.test[0].shape[1]*lstnet_init.mc_iterations)))
            MC_pred_mean = np.mean(Yt_hat, 0)
            #MC_pred_median = np.median(Yt_hat, 0)
            MC_pred_std = np.std(Yt_hat, 0)
            # reshape Yt_hat = Yt_hat.reshape(Yt_hat.shape[1]*Yt_hat.shape[0], Yt_hat.shape[2])
            # pd.DataFrame(Yt_hat).to_csv(lstnet_init.save + "_Y_hat.csv", index=False)
            #rescale ? Yt_hat = Yt_hat * self.std_y_train + self.mean_y_train
            # MC_pred = np.mean(Yt_hat, 0)
            pd.DataFrame(testPredict).to_csv(lstnet_init.save + "_Y_hat.csv", index=False)
            pd.DataFrame(Data.test[1]).to_csv(lstnet_init.save + "_Y.csv", index=False)
            pd.DataFrame(MC_pred_mean).to_csv(lstnet_init.save + "_Y_hat_mean.csv", index=False)
            #pd.DataFrame(MC_pred_median).to_csv(lstnet_init.save + "_Y_hat_median.csv", index=False)
            pd.DataFrame(MC_pred_std).to_csv(lstnet_init.save + "_Y_hat_std.csv", index=False)

            ## TODO: add ro risk estimates

        else:
            testPredict = None

        if lstnet_init.plot is True:
            #PlotPrediction(Data, lstnet_init, trainPredict, validPredict, testPredict)
            PlotPrediction(Data, lstnet_init, trainPredict, validPredict, MC_pred_mean)
