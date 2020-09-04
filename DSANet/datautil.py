import numpy as np
import pandas as pd
import datetime as dt
from sklearn import preprocessing

class DataUtil(object):
    #
    # This class contains data specific information.
    # It does the following:
    #  - Read data from file
    #  - Normalise it
    #  - Split it into train, dev (validation) and test
    #  - Create X and Y for each of the 3 sets (train, dev, test) according to the following:
    #    Every sample (x, y) shall be created as follows:
    #     - x --> window number of values
    #     - y --> one value that is at horizon in the future i.e. that is horizon away past the last value of x
    #    This way X and Y will have the following dimensions:
    #     - X [number of samples, window, number of multivariate time series]
    #     - Y [number of samples, number of multivariate time series]
    
    def __init__(self, hparams, normalise = 2):
        try:
            self.hparams = hparams
            
            # self.hparams.data_name, , 
            #print("Start reading data")
            filename = './data/{}/{}{}'.format(self.hparams.data_name, self.hparams.data_name, '.csv')
            self.rawdata = pd.read_csv(filename, parse_dates=[0])
            index_colname = self.rawdata.columns[0]
            series = {'all':[0,-1], 'load':[0,59], 'price':[59,90], 'wind':[90, 147], 'solar':[147, 183]}
            splits = self.datasplitter(self.hparams.powerset)
            if self.hparams.n_multiv == 183 or self.hparams.n_multiv == 321:
                self.rawdata.drop([index_colname], axis=1, inplace=True)
                self.rawdata = self.rawdata.to_numpy()
                if splits != 0:
                    self.rawdata = self.rawdata[:,splits[0]:splits[1]]
            else:
                if splits != 0:
                    if splits[0] != 0:
                        self.rawdata = self.rawdata.iloc[:, np.r_[0,splits[0]+1:splits[1]+1]]
                    else:
                        self.rawdata = self.rawdata.iloc[:, splits[0]:splits[1]+1]
                if self.hparams.calendar == 'True' or self.hparams.calendar == 1 or self.hparams.n_multiv == 189 or self.hparams.n_multiv == 327:
                    data = self.rawdata
                    data['hour'] = data[index_colname].dt.hour
                    data = self.encode(data, 'hour', 23)
                    data['month'] = data[index_colname].dt.month
                    data = self.encode(data, 'month', 12)
                    data['day'] = data[index_colname].dt.day
                    data = self.encode(data, 'day', 365)
                    self.rawdata = data
                    self.rawdata.drop([index_colname, 'hour', 'month', 'day'], axis=1, inplace=True)
                else:
                    self.rawdata.drop([index_colname], axis=1, inplace=True)

                self.rawdata = self.rawdata.to_numpy()


            #print("End reading data")

            self.w         = self.hparams.window
            self.h         = self.hparams.horizon
            self.data      = np.zeros(self.rawdata.shape)
            self.trainpart = self.hparams.split_train
            self.n, self.m = self.data.shape
            self.normalise = normalise
            self.normalise_data(normalise)
            self.split_data(self.hparams.split_train, self.hparams.split_validation, self.hparams.split_test)
        except IOError as err:
            # In case file is not found, all of the above attributes will not have been created
            # Hence, in order to check if this call was successful, you can call hasattr on this object 
            # to check if it has attribute 'data' for example
            print(f"Error opening data file ... {err}")
        
        
    def normalise_data(self, normalise):
        #print(f"Normalise: {normalise}")

        if normalise == 0: # do not normalise
            self.data = self.rawdata
        
        if normalise == 1: # normalise each timeseries alone. This is the default mode
            self.scale = np.ones(self.m)
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdata[:int(self.trainpart * self.n), i]))
                self.data[:, i] = self.rawdata[:, i] / self.scale[i]

        if normalise == 2: # normalise timeseries using MinMaxScaler
            # MinMaxScaler RobustScaler Normalizer MaxAbsScaler StandardScaler
            self.scaler = preprocessing.MinMaxScaler()
            self.scale = self.scaler.fit(self.rawdata[:int(self.trainpart * self.n)])
            self.data = self.scale.transform(self.rawdata)

    
    def split_data(self, train, valid, test):
        #print(f"Splitting data into training set ({train}), validation set ({valid}) and testing set ({1 - (train + valid)})")

        train_set = range(self.w + self.h - 1, int(train * self.n))
        valid_set = range(int(train * self.n), int((train + valid) * self.n))
        if test  == 0.001: # the default value --> we use rest for testing
            test_set  = range(int((train + valid) * self.n), self.n)
        else: # if test set vas defined we used the portion after validation to test
            test_set  = range(int((train + valid) * self.n), int((train + valid + test) * self.n))


        
        self.train = self.get_data(train_set)
        self.valid = self.get_data(valid_set)
        self.test  = self.get_data(test_set)
        
    def get_data(self, rng):
        n = len(rng)
        
        X = np.zeros((n, self.w, self.m))
        Y = np.zeros((n, 1, self.m))
        
        for i in range(n):
            end   = rng[i] - self.h + 1
            start = end - self.w
            
            X[i,:,:] = self.data[start:end, :]
            Y[i,:,:] = self.data[rng[i],:]
        
        #print(f"Shape of data X: {X.shape}, Y: {Y.shape} ")
        return [X, Y]

    def encode(self, data, col, max_val):
        data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
        data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
        return data

    def datasplitter(self,i):
        switcher={
                'load':[0,59],
                'price':[59,90],
                'wind':[90,147],
                'solar':[147,183],
             }
        return switcher.get(i,0)

    def __len__(self):
        if self.set_type == 'train':
            return self.sample_num_train
        elif self.set_type == 'validation':
            return self.sample_num_validation
        else:
            return self.sample_num_test

    def __getitem__(self, idx):
        sample = [self.samples[idx, :, :], self.labels[idx, :, :]]

        return sample


