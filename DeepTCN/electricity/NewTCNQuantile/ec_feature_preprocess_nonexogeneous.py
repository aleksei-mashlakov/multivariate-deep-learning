import os, math
import _pickle as pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn import preprocessing

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', default='data', help='Parent dir of the dataset')
parser.add_argument('--file-name', default='electricity.csv', help='Directory containing data.csv')
parser.add_argument('--pickle-name', default='electricity.pkl', help='Directory containing data.csv')
parser.add_argument('--horizon', type=int, default=24, help='Forecast horizon. Default=24')
parser.add_argument('--test', action='store_true', help='whenever to use test set only.')
parser.add_argument('--hop', action='store_true', help='Whether to use test set for validation') # default=False


if __name__ == '__main__':
    
    args = parser.parse_args()
    
    ### load the data
    dir_path = args.data_folder # './data'
    file_name = args.file_name
    
    if file_name=='electricity.csv':
        train_start = '2012-01-01 00:00:00'
        if args.test: 
            train_end = '2013-10-19 23:00:00'
            test_start = '2014-05-20 00:00:00' #need additional 7 days as given info 
            test_end = '2014-12-31 23:00:00'
        elif args.hop:
            train_end = '2012-04-30 23:00:00'
            test_start = '2012-04-24 00:00:00'
            test_end = '2012-05-31 23:00:00'
        else:
            train_end = '2013-10-19 23:00:00'
            test_start = '2013-10-20 00:00:00' #need additional 7 days as given info 
            test_end = '2014-12-31 23:00:00' 
            
    elif file_name=='europe_power_system.csv':
        train_start = '2015-01-01 00:00:00'
        if args.test:
            train_end = '2017-01-15 23:00:00'
            test_start = '2017-06-17 00:00:00' #need additional 7 days as given info
            test_end = '2017-11-30 23:00:00'
        elif args.hop:
            train_end = '2015-04-30 23:00:00' 
            test_start = '2015-04-24 00:00:00' #need additional 7 days as given info 
            test_end = '2015-05-31 23:00:00'
        else:
            train_end = '2017-01-15 23:00:00'
            test_start = '2017-01-16 00:00:00' #need additional 7 days as given info
            test_end = '2017-11-30 23:00:00'
            
    
    df = pd.read_csv(os.path.join(dir_path, file_name), sep=",", index_col=0, parse_dates=True, decimal='.')
    df = df.reset_index()
    df = df.drop([df.columns[0]], axis=1).transpose()
    dt = df.rename(columns=df.iloc[0]).values #.drop(df.index[0])
    
    ## The date range
    date_list = pd.date_range(start=train_start, end=test_end)
    date_list = pd.to_datetime(date_list)
    yr = int(date_list.year[0])

    hour_list = []
    for nDate in date_list:
        for nHour in range(24):
            tmp_timestamp = nDate+timedelta(hours=nHour)
            hour_list.append(tmp_timestamp)
    hour_list = np.array(hour_list)
    #print('hour_list', hour_list.shape[0])
    #print('dt.shape[0]', dt.shape[0])


    station_index = list(range(dt.shape[0]))
    #if args.horizon ==36:
    #    sliding_window_dis = 24;
    #else:
    #    sliding_window_dis = args.horizon;
    #print('sliding_window_dis: ', sliding_window_dis)
    sliding_window_dis = args.horizon; # 24;
    input_len = 168;
    output_len = args.horizon; #24;
    sample_len = input_len + output_len; #192; #168+24
    
    coef = args.horizon/24;

    total_n = int((len(date_list) - 8)/coef) #800; ## The total days
    test_n = int(len(pd.date_range(start=test_start, end=test_end))/coef) #7     ## The testing days, day of the last 7 days
    train_n = total_n - test_n ## The training days
    #print('train_n', train_n)
    #print('test_n', test_n)

    trainX_list = [];trainX2_list = [];trainY_list = [];trainY2_list = []
    testX_list = [];testX2_list = [];testY_list = [];testY2_list = []
    #for station in station_index:
    for station in station_index:
        print('Station', station)
        sub_series = dt[station,1:].astype('float32')
        sub_index = np.array(range(dt.shape[1]-1))-np.min(np.where(sub_series>0))
        trainX = np.zeros(shape=(train_n, input_len))       ## The input series
        trainY = np.zeros(shape=(train_n, output_len))      ## The output series  

        testX  = np.zeros(shape=(test_n, input_len))        ## The input series
        testY = np.zeros(shape=(test_n, output_len))        ## The output series
        #print('Shape testX, testY: ', testX.shape, testY.shape)
        #print('Shape trainX, trainY: ', trainX.shape, trainY.shape)

        covariate_num = 2 #8   # other features covariate_num: sub_index, station_id,nYear,nMonth,day_of_month, day_of_week, iHour
        trainX2 = np.zeros(shape=(train_n, input_len, covariate_num))
        trainY2 = np.zeros(shape=(train_n, output_len, covariate_num))
        testX2 = np.zeros(shape=(test_n, input_len, covariate_num))
        testY2 = np.zeros(shape=(test_n, output_len, covariate_num))
        ### Testing samples (7+1)*24
        ts_len = sub_series.shape[0]
        #print('Ts_len: ', ts_len)
        if args.hop:
            start_index = hour_list.shape[0]-sample_len
        else:
            start_index = ts_len-sample_len
        #print('start_index: ', start_index)
        #print('total_n', total_n)
        for i in range(total_n):
            ### The sequence data
            series_x = sub_series[start_index:start_index+input_len]
            series_y = sub_series[start_index+input_len:start_index+sample_len]
            #print('series_x', series_x)
            ### The index data
            hour_mean = np.mean(series_x.reshape(-1,24),axis=0)
            index_x = np.tile(hour_mean,7)
            index_y = np.tile(hour_mean, 2)[:args.horizon]
            
            #index_y = np.tile(hour_mean,1)
            ### The covariate
            station_X = np.repeat(station, input_len)
            station_Y = np.repeat(station, output_len)
            ### the time index
#             time_index_x = pd.to_datetime(hour_list[start_index:start_index+input_len])
            #print('start_index+input_len', start_index+input_len)
            #print('start_index+sample_len', start_index+sample_len)
#             time_index_y = pd.to_datetime(hour_list[start_index+input_len:start_index+sample_len])
#             nYear_X, nYear_Y = time_index_x.year-yr, time_index_y.year-yr
#             nMonth_X, nMonth_Y = time_index_x.month-1, time_index_y.month-1
#             mDay_X, mDay_Y = time_index_x.day-1, time_index_y.day-1
#             wDay_X, wDay_Y = time_index_x.weekday, time_index_y.weekday
            #print('wDay_X, wDay_Y', wDay_X, wDay_Y)

#             nHour_X, nHour_Y = time_index_x.hour, time_index_y.hour
#             holiday_X, holiday_Y = (wDay_X>=5),(wDay_Y>=5)
            #print('holiday_X, holiday_Y', holiday_X, holiday_Y)
            #print(station_X,index_x,nYear_X,nMonth_X,mDay_X,wDay_X,nHour_X,holiday_X)

            covariate_X = np.c_[station_X,index_x]#,nYear_X,nMonth_X,mDay_X,wDay_X,nHour_X,holiday_X]
            covariate_Y = np.c_[station_Y,index_y]#,nYear_Y,nMonth_Y,mDay_Y,wDay_Y,nHour_Y,holiday_Y]

            if(i<test_n):
                test_index = i
                testX[test_index] = series_x
                testY[test_index] = series_y
                testX2[test_index] = covariate_X
                testY2[test_index] = covariate_Y

            else:
                trainX[i-test_n] = series_x
                trainY[i-test_n] = series_y
                trainX2[i-test_n] = covariate_X
                trainY2[i-test_n] = covariate_Y
            # update the start_index
            start_index = start_index - sliding_window_dis
            #print('start_index',start_index)


        testX_list.append(testX)
        testX2_list.append(testX2)
        testY_list.append(testY)
        testY2_list.append(testY2)

        trainX_list.append(trainX)
        trainX2_list.append(trainX2)
        trainY_list.append(trainY)
        trainY2_list.append(trainY2)


    testX_dt = np.vstack(testX_list)
    testY_dt = np.vstack(testY_list)
    testX2_dt = np.vstack(testX2_list)
    testY2_dt = np.vstack(testY2_list)

    trainX_dt = np.vstack(trainX_list)
    trainY_dt = np.vstack(trainY_list)
    trainX2_dt = np.vstack(trainX2_list)
    trainY2_dt = np.vstack(trainY2_list)

    scaler = preprocessing.StandardScaler()
    scaler.fit(trainX_dt)

    trainX_dt = scaler.transform(trainX_dt)
    testX_dt = scaler.transform(testX_dt)
    scaler2 = preprocessing.StandardScaler()
    scaler2.fit(trainX2[:,:,1])
    trainX2[:,:,1] = scaler2.transform(trainX2[:,:,1])
    testX2[:,:,1] = scaler2.transform(testX2[:,:,1])

    ### The filter data
    ### Select the data of the Nov
#     isNov = trainX2_dt[:,:,3]>=11
#     trainX_dt = trainX_dt[isNov[:,0]]
#     trainY_dt = trainY_dt[isNov[:,0]]
#     trainX2_dt = trainX2_dt[isNov[:,0]]
#     trainY2_dt = trainY2_dt[isNov[:,0]]

    ### Save the data
    with open(os.path.join(dir_path, args.pickle_name), 'wb') as f:
        pickle.dump([trainX_dt,trainX2_dt, trainY_dt,trainY2_dt, testX_dt, testX2_dt,testY_dt,testY2_dt], f, -1)
        
    with open(os.path.join(dir_path, 'scaler_' + args.pickle_name), 'wb') as f:
        pickle.dump([scaler, scaler2], f, -1)
