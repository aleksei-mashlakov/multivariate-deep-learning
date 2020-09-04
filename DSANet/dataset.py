import torch
import torch.utils.data
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime


class MTSFDataset(torch.utils.data.Dataset):
    """Multi-variate Time-Series Dataset for *.txt file

    Returns:
        [sample, label]
    """

    def __init__(self,
                 window,
                 horizon,
                 data_name='wecar',
                 set_type='train',  # 'train'/'validation'/'test'
                 data_dir='./data',
                 split_train=0.6,
                 split_validation=0.2,
                 split_test=0.2, ):
        assert type(set_type) == type('str')
        self.window = window
        self.horizon = horizon
        self.data_dir = data_dir
        self.set_type = set_type
        split = [0, split_train, split_train + split_validation, split_train + split_validation + split_test]
        self.set_split = split

        file_path = os.path.join(data_dir, data_name, '{}.txt'.format(data_name))
        rawdata = np.loadtxt(open(file_path), delimiter=',')
        self.len, self.var_num = rawdata.shape

        self.sample_num_train = max(int(self.len * self.set_split[1]) - self.window - self.horizon + 1, 0)
        self.sample_num_validation = max(int(self.len * (self.set_split[2] - self.set_split[1])) - self.window - self.horizon + 1, 0)
        self.sample_num_test = max(int(self.len * (self.set_split[3] - self.set_split[2])) - self.window - self.horizon + 1, 0)

        self.samples, self.labels = self.__getsamples(rawdata)

    def __getsamples(self, data):

        if self.set_type == 'train':
            X = torch.zeros((self.sample_num_train, self.window, self.var_num))
            Y = torch.zeros((self.sample_num_train, 1, self.var_num))

            for i in range(self.sample_num_train):
                start = i
                end = i + self.window
                X[i, :, :] = torch.from_numpy(data[start:end, :])
                Y[i, :, :] = torch.from_numpy(data[end + self.horizon - 1, :])
        elif self.set_type == 'validation':
            X = torch.zeros((self.sample_num_validation, self.window, self.var_num))
            Y = torch.zeros((self.sample_num_validation, 1, self.var_num))

            for i in range(self.sample_num_validation):
                start = i + self.sample_num_train
                end = i + self.sample_num_train + self.window
                X[i, :, :] = torch.from_numpy(data[start:end, :])
                Y[i, :, :] = torch.from_numpy(data[end + self.horizon - 1, :])
        else:
            X = torch.zeros((self.sample_num_test, self.window, self.var_num))
            Y = torch.zeros((self.sample_num_test, 1, self.var_num))

            for i in range(self.sample_num_test):
                start = i + self.sample_num_train + self.sample_num_validation
                end = i + self.sample_num_train + self.sample_num_validation + self.window
                X[i, :, :] = torch.from_numpy(data[start:end, :])
                Y[i, :, :] = torch.from_numpy(data[end + self.horizon - 1, :])

        # print(f"{self.set_type}: {start}, {end}, {X.shape}, {Y.shape}")
        return X, Y

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


# test
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    logging.debug('Test: data from .txt file')
    dataset = MTSFDataset(
        window=16,
        horizon=3,
        data_name='wecar',
        set_type='train',
        data_dir='./data'
    )
    i = 0
    sample = dataset[i]
    logging.debug('Sample #{}, label: {}'.format(i, sample[1]))
    logging.debug('data: {}'.format(sample[0]))
