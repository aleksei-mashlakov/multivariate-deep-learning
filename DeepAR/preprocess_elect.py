from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime, timedelta
import pandas as pd
import math
import numpy as np
import random
from tqdm import trange

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

from math import sqrt
from pandas import read_csv, DataFrame
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='elect', help='Name of the dataset')
parser.add_argument('--data-folder', default='data', help='Parent dir of the dataset')
parser.add_argument('--model-name', default='base_model_24', help='Directory containing params.json')
parser.add_argument('--file-name', default='electricity.csv', help='Directory containing data.csv')
parser.add_argument('--test', action='store_true', help='Whether to use test set for validation') # default=False
parser.add_argument('--hop', action='store_true', help='Whether to use test set for validation') # default=False



def prep_data(data, covariates, data_start, train = True):
    #print("train: ", train)
    time_len = data.shape[0]
    #print("time_len: ", time_len)
    input_size = window_size-stride_size
    windows_per_series = np.full((num_series), (time_len-input_size) // stride_size)
    #print("windows pre: ", windows_per_series.shape)
    if train: windows_per_series -= (data_start+stride_size-1) // stride_size
    #print("data_start: ", data_start.shape)
    #print(data_start)
    #print("windows: ", windows_per_series.shape)
    #print(windows_per_series)
    total_windows = np.sum(windows_per_series)
    x_input = np.zeros((total_windows, window_size, 1 + num_covariates + 1), dtype='float32')
    label = np.zeros((total_windows, window_size), dtype='float32')
    v_input = np.zeros((total_windows, 2), dtype='float32')
    #cov = 3: ground truth + age + day_of_week + hour_of_day + num_series
    #cov = 4: ground truth + age + day_of_week + hour_of_day + month_of_year + num_series
    count = 0
    if not train:
        covariates = covariates[-time_len:]
    for series in trange(num_series):
        cov_age = stats.zscore(np.arange(total_time-data_start[series]))
        if train:
            covariates[data_start[series]:time_len, 0] = cov_age[:time_len-data_start[series]]
        else:
            covariates[:, 0] = cov_age[-time_len:]
        for i in range(windows_per_series[series]):
            if train:
                window_start = stride_size*i+data_start[series]
            else:
                window_start = stride_size*i
            window_end = window_start+window_size
            '''
            print("x: ", x_input[count, 1:, 0].shape)
            print("window start: ", window_start)
            print("window end: ", window_end)
            print("data: ", data.shape)
            print("d: ", data[window_start:window_end-1, series].shape)
            '''
            x_input[count, 1:, 0] = data[window_start:window_end-1, series]
            x_input[count, :, 1:1+num_covariates] = covariates[window_start:window_end, :]
            x_input[count, :, -1] = series
            label[count, :] = data[window_start:window_end, series]
            nonzero_sum = (x_input[count, 1:input_size, 0]!=0).sum()
            if nonzero_sum == 0:
                v_input[count, 0] = 0
            else:
                v_input[count, 0] = np.true_divide(x_input[count, 1:input_size, 0].sum(),nonzero_sum)+1
                x_input[count, :, 0] = x_input[count, :, 0]/v_input[count, 0]
                if train:
                    label[count, :] = label[count, :]/v_input[count, 0]
            count += 1
    prefix = os.path.join(save_path, 'train_' if train else 'test_')
    np.save(prefix+'data_'+save_name, x_input)
    np.save(prefix+'v_'+save_name, v_input)
    np.save(prefix+'label_'+save_name, label)

def gen_covariates(times, num_covariates):
    covariates = np.zeros((times.shape[0], num_covariates))
    for i, input_time in enumerate(times):
        covariates[i, 1] = input_time.weekday()
        covariates[i, 2] = input_time.hour
        covariates[i, 3] = input_time.month
    for i in range(1,num_covariates):
        covariates[:,i] = stats.zscore(covariates[:,i])
    return covariates[:, :num_covariates]

def visualize(data, week_start):
    x = np.arange(window_size)
    f = plt.figure()
    plt.plot(x, data[week_start:week_start+window_size], color='b')
    f.savefig("visual.png")
    plt.close()

if __name__ == '__main__':

    global save_path
    # Load the parameters from json file
    args = parser.parse_args()
    model_dir = os.path.join('experiments', args.model_name)
    json_path = os.path.join(model_dir, 'params.json')
    #data_dir = os.path.join('data', args.dataset)
    assert os.path.isfile(json_path), f'No json configuration file found at {json_path}'
    params = utils.Params(json_path)


    name = args.file_name #'electricity.csv'
    save_name = args.dataset #'elect'
    window_size = params.train_window #192
    stride_size = params.predict_steps #24
    num_covariates = 4

    #test = False
    # print('args.test: ', args.test)

    if not args.test:
        train_start = '2012-01-01 00:00:00'
        train_end = '2013-10-19 23:00:00' #'2014-06-30 23:00:00'
        test_start = '2013-10-13 00:00:00' #need additional 7 days as given info  '2014-06-24 00:00:00'
        test_end = '2014-05-26 23:00:00' #2014-12-28
    else:
        train_start = '2012-01-01 00:00:00'
        train_end = '2013-10-19 23:00:00' #'2014-06-30 23:00:00'
        test_start = '2014-05-20 00:00:00' #need additional 7 days as given info  '2014-06-24 00:00:00'
        test_end = '2014-12-31 23:00:00' #2014-12-28

    if args.hop:
        train_start = '2012-01-01 00:00:00'
        train_end = '2012-04-30 23:00:00' #'2014-06-30 23:00:00'
        test_start = '2012-04-24 00:00:00' #need additional 7 days as given info  '2014-06-24 00:00:00'
        test_end = '2012-05-31 23:00:00' #2014-12-28

    pred_days = 7
    given_days = 7

    save_path = model_dir # os.path.join('data', save_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    csv_path = os.path.join(args.data_folder, save_name, name) #os.path.join(save_path, name)

    data_frame = pd.read_csv(csv_path, sep=",", index_col=0, parse_dates=True, decimal=',').astype(float)
    #data_frame = data_frame.resample('1H',label = 'left',closed = 'right').sum()[train_start:test_end]
    data_frame.fillna(0, inplace=True)
    # data_start = (data_frame!=0).argmax(axis=0) #find first nonzero value in each time series ########### added
    #data_frame = data_frame.drop(data_frame.columns[data_start>=161], axis=1) ########### added
    covariates = gen_covariates(data_frame[train_start:test_end].index, num_covariates)
    train_data = data_frame[train_start:train_end].values
    test_data = data_frame[test_start:test_end].values
    data_start = (train_data!=0).argmax(axis=0) #find first nonzero value in each time series
    total_time = data_frame.shape[0] #32304
    num_series = data_frame.shape[1] #370
    prep_data(train_data, covariates, data_start)
    prep_data(test_data, covariates, data_start, train=False)
