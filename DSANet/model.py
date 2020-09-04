import numpy as np
import logging as log
from collections import OrderedDict
import torch.nn as nn
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

import pytorch_lightning as ptl
from pytorch_lightning import LightningModule
from dataset import MTSFDataset
from datautil import DataUtil

from dsanet.Layers import EncoderLayer, DecoderLayer


class Single_Global_SelfAttn_Module(nn.Module):

    def __init__(
            self, hparams):
        """.
		Args:

		window (int): the length of the input window size
		n_multiv (int): num of univariate time series
		n_kernels (int): the num of channels
		w_kernel (int): the default is 1
		d_k (int): d_model / n_head
		d_v (int): d_model / n_head
		d_model (int): outputs of dimension
		d_inner (int): the inner-layer dimension of Position-wise Feed-Forward Networks
		n_layers (int): num of layers in Encoder
		n_head (int): num of Multi-head
		drop_prob (float): the probability of dropout
		"""

        super(Single_Global_SelfAttn_Module, self).__init__()
        self.hparams = hparams
        self.window = hparams.window
        self.w_kernel = hparams.w_kernel
        self.n_multiv = hparams.n_multiv
        self.d_model = hparams.d_model
        self.drop_prob = hparams.drop_prob
        self.conv2 = nn.Conv2d(1, hparams.n_kernels, (hparams.window, hparams.w_kernel))
        self.in_linear = nn.Linear(hparams.n_kernels, hparams.d_model)
        self.out_linear = nn.Linear(hparams.d_model, hparams.n_kernels)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(hparams)
            for _ in range(hparams.n_layers)])

    def forward(self, x, return_attns=False):
        self.n_multiv = x.shape[2]
        x = x.view(-1, self.w_kernel, self.window, self.n_multiv)
        x2 = F.relu(self.conv2(x))
        if self.hparams.mcdropout == 'True':
            x2 = nn.functional.dropout(x2, p=self.drop_prob, training=True)
        else:
            x2 = nn.Dropout(p=self.drop_prob)(x2)
        #x2 = nn.Dropout(p=self.drop_prob)(x2)
        x = torch.squeeze(x2, 2)
        x = torch.transpose(x, 1, 2)
        src_seq = self.in_linear(x)

        enc_slf_attn_list = []

        enc_output = src_seq

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        enc_output = self.out_linear(enc_output)
        return enc_output,


class Single_Local_SelfAttn_Module(nn.Module):

    def __init__(
            self, hparams):
        """
		Args:

		window (int): the length of the input window size
		n_multiv (int): num of univariate time series
		n_kernels (int): the num of channels
		w_kernel (int): the default is 1
		d_k (int): d_model / n_head
		d_v (int): d_model / n_head
		d_model (int): outputs of dimension
		d_inner (int): the inner-layer dimension of Position-wise Feed-Forward Networks
		n_layers (int): num of layers in Encoder
		n_head (int): num of Multi-head
		drop_prob (float): the probability of dropout
		"""

        super(Single_Local_SelfAttn_Module, self).__init__()
        self.hparams = hparams
        self.window = hparams.window
        self.w_kernel = hparams.w_kernel
        self.n_multiv = hparams.n_multiv
        self.d_model = hparams.d_model
        self.drop_prob = hparams.drop_prob
        self.conv1 = nn.Conv2d(1, hparams.n_kernels, (hparams.local, hparams.w_kernel))
        self.pooling1 = nn.AdaptiveMaxPool2d((1, hparams.n_multiv))
        self.in_linear = nn.Linear(hparams.n_kernels, hparams.d_model)
        self.out_linear = nn.Linear(hparams.d_model, hparams.n_kernels)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(hparams)
            for _ in range(hparams.n_layers)])

    def forward(self, x, return_attns=False):
        self.n_multiv = x.shape[2]
        x = x.view(-1, self.w_kernel, self.window, self.n_multiv)
        x1 = F.relu(self.conv1(x))
        x1 = self.pooling1(x1)
        if self.hparams.mcdropout == 'True':
            output = nn.functional.dropout(x1, p=self.drop_prob, training=True)
        else:
            x1 = nn.Dropout(p=self.drop_prob)(x1)

        #x1 = nn.Dropout(p=self.drop_prob)(x1)
        x = torch.squeeze(x1, 2)
        x = torch.transpose(x, 1, 2)
        src_seq = self.in_linear(x)

        enc_slf_attn_list = []

        enc_output = src_seq

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        enc_output = self.out_linear(enc_output)
        return enc_output,


class AR(nn.Module):

    def __init__(self, window):
        super(AR, self).__init__()
        self.linear = nn.Linear(window, 1)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.linear(x)
        x = torch.transpose(x, 1, 2)
        return x


class DSANet(LightningModule):

    def __init__(self, hparams):
        """
		Pass in parsed HyperOptArgumentParser to the model
		"""
        super(DSANet, self).__init__()
        self.hparams = hparams
        self.test_results = None
        self.val_results = None
        self.batch_size = hparams.batch_size

        # parameters from dataset
        self.window = hparams.window
        self.local = hparams.local
        self.n_multiv = hparams.n_multiv
        self.n_kernels = hparams.n_kernels
        self.w_kernel = hparams.w_kernel
        if hasattr(hparams, 'scale'):
            self.scale = hparams.scale
        else:
            self.scale = None
        if hasattr(hparams, 'scaler'):    
            self.scaler = hparams.scaler
        else:
            self.scaler = None

        # hyperparameters of model
        self.d_model = hparams.d_model
        self.d_inner = hparams.d_inner
        self.n_layers = hparams.n_layers
        self.n_head = hparams.n_head
        self.d_k = hparams.d_k
        self.d_v = hparams.d_v
        self.drop_prob = hparams.drop_prob

        # build model
        self.__build_model()

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        """
		Layout model
		"""
        self.sgsf = Single_Global_SelfAttn_Module(self.hparams)

        self.slsf = Single_Local_SelfAttn_Module(self.hparams)

        self.ar = AR(window=self.window)
        self.W_output1 = nn.Linear(2 * self.n_kernels, 1)
        self.dropout = nn.Dropout(p=self.drop_prob)
        self.active_func = nn.Tanh()

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        """
		No special modification required for lightning, define as you normally would
		"""
        #print(f'In forward: testing = {str(self.trainer.testing)}, training = {str(self.trainer.model.training)}')
        sgsf_output, *_ = self.sgsf(x)
        slsf_output, *_ = self.slsf(x)
        sf_output = torch.cat((sgsf_output, slsf_output), 2)
        # sf_output = self.dropout(sf_output)
        if self.hparams.mcdropout == 'True':
            sf_output = nn.functional.dropout(sf_output, p=self.drop_prob, training=True)
        else:
            sf_output = self.dropout(sf_output)
        sf_output = self.W_output1(sf_output)
        sf_output = torch.transpose(sf_output, 1, 2)

        ar_output = self.ar(x)

        logits = sf_output + ar_output
       
        return logits

    def loss(self, labels, logits):
        # batch_size, horizon, variable_num
        if self.hparams.criterion == 'l1_loss':
            loss = F.l1_loss(logits, labels)
        elif self.hparams.criterion == 'mse_loss':
            loss = F.mse_loss(logits, labels)
        return loss

    def training_step(self, batch, batch_idx):
        """
		Lightning calls this inside the training loop
		"""
        # forward pass
        x, y = batch
        #x = x.view(x.size(0), -1)

        y_hat = self.forward(x)

        # calculate loss
        loss_val = self.loss(y, y_hat)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
        
        tqdm_dict = {'train_loss': loss_val}
        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            #'log': tqdm_dict
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output
    
    def validation_step(self, batch, batch_idx):
        """
		Lightning calls this inside the validation loop
		"""
        #print('validation_step: We are validating..')
        x, y = batch
        #x = x.view(x.size(0), -1)
        y_hat = self.forward(x)

        loss_val = self.loss(y, y_hat)

        local_hparams = self.hparams
        if hasattr(local_hparams, 'scaler'):
            squeezed_y = torch.squeeze(y).cpu()
            squeezed_y_hat = torch.squeeze(y_hat.detach().cpu())
            if self.on_gpu:
                y_inversed = torch.tensor(self.dataset.scaler.inverse_transform(torch.squeeze(y).cpu())).cuda(x.device.index)
                y_hat_inversed = torch.tensor(self.dataset.scaler.inverse_transform(torch.squeeze(y_hat.detach().cpu()))).cuda(x.device.index)
            else: 
                y_inversed = torch.tensor(self.scaler.inverse_transform(torch.squeeze(y).cpu()))
                y_hat_inversed = torch.tensor(self.scaler.inverse_transform(torch.squeeze(y_hat.detach().cpu())))

            y_inversed = torch.unsqueeze(y_inversed, 1)
            y_hat_inversed = torch.unsqueeze(y_hat_inversed, 1)
        elif hasattr(local_hparams, 'scale'):
            if self.on_gpu:
                y_inversed = torch.squeeze(y) * torch.tensor(self.dataset.scale).cuda(x.device.index)
                y_hat_inversed = torch.squeeze(y_hat) * torch.tensor(self.dataset.scale).cuda(x.device.index)
            else:
                y_inversed = torch.squeeze(y) * torch.tensor(self.dataset.scale)
                y_hat_inversed = torch.squeeze(y_hat) * torch.tensor(self.dataset.scale)

            y_inversed = torch.unsqueeze(y_inversed, 1)
            y_hat_inversed = torch.unsqueeze(y_hat_inversed, 1)
        else:
            y_inversed = y
            y_hat_inversed = y_hat

        unscaled_loss_val = self.loss(y_inversed, y_hat_inversed)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
            unscaled_loss_val = unscaled_loss_val.unsqueeze(0)

        output = OrderedDict({
            'val_loss': loss_val,
            'y': y,
            'y_hat': y_hat,
            'y_inversed': y_inversed,
            'y_hat_inversed': y_hat_inversed,
            'unscaled_loss': unscaled_loss_val
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_epoch_end(self, outputs):
        """
		Called at the end of validation to aggregate outputs
		:param outputs: list of individual outputs of each validation step
		"""
        #print('validation_epoch_end: We are collecting validation results..')
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        val_loss_mean = 0
        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

        val_loss_mean /= len(outputs)


        y = torch.cat(([x['y_inversed'] for x in outputs]), 0)
        y_hat = torch.cat(([x['y_hat_inversed'] for x in outputs]), 0)

        num_var = y.size(-1)
        y = y.view(-1, num_var)
        y_hat = y_hat.view(-1, num_var)
        sample_num = y.size(0)
        
        # Save original values
        y_hat_org= y_hat
        y_org = y

        val_rrse={}
        val_rse={}
        val_nrmse={}
        val_nd={}
        val_rho10={}
        val_rho50={}
        val_rho90={}
        val_corr={}
        val_mae={}
        val_maape={}

        if self.hparams.calendar == 'True' or self.hparams.n_multiv == 189 or self.hparams.n_multiv == 327:
            series = {'all':[0,-6], 'load':[0,59], 'price':[59,90], 'wind':[90, 147], 'solar':[147, 183]}
        else:
            series = {'all':[0,None], 'load':[0,59], 'price':[59,90], 'wind':[90, 147], 'solar':[147, 183]}

        if self.hparams.data_name == 'europe_power_system' and self.hparams.powerset == 'all':
            for key in series:
                y_hat = y_hat_org[:,series[key][0]:series[key][1]]
                y = y_org[:,series[key][0]:series[key][1]]
                num_var = y.size(-1)
                sample_num = y.size(0)
                y_diff = y_hat - y
                y_mean = torch.mean(y)
                y_translation = y - y_mean

                val_maape[key] = (torch.mean(torch.atan(torch.abs((y-y_hat) / y)))).item()
                val_rrse[key] = (torch.sqrt(torch.sum(torch.pow(y_diff, 2))) / torch.sqrt(torch.sum(torch.pow(y_translation, 2)))).item()
                val_rse[key] = (torch.mean(torch.pow(y_diff, 2)) / torch.sum(torch.pow(y_translation, 2))).item()
                val_nrmse[key] = (torch.sqrt(torch.mean(torch.pow(y_diff, 2))) / torch.mean(torch.abs(y))).item()
                val_nd[key] = (torch.sum(torch.abs(y_diff)) / torch.sum(torch.abs(y))).item()

                # val_rho10', 'val_rho50', 'val_rho90'
                val_rho10[key] = (2 * (torch.sum((y - y_hat) * 0.1 * (y >= y_hat).float()) + torch.sum(
                    (y_hat - y) * (1 - 0.1) * (y < y_hat).float())) / torch.sum(torch.abs(y))).item()
                val_rho50[key] = (2 * (torch.sum((y - y_hat) * 0.5 * (y >= y_hat).float()) + torch.sum(
                    (y_hat - y) * (1 - 0.5) * (y < y_hat).float())) / torch.sum(torch.abs(y))).item()
                val_rho90[key] = (2 * (torch.sum((y - y_hat) * 0.9 * (y >= y_hat).float()) + torch.sum(
                    (y_hat - y) * (1 - 0.9) * (y < y_hat).float())) / torch.sum(torch.abs(y))).item()

                y_m = torch.mean(y, 0, True)
                y_hat_m = torch.mean(y_hat, 0, True)
                y_d = y - y_m
                y_hat_d = y_hat - y_hat_m
                corr_top = torch.sum(y_d * y_hat_d, 0)
                corr_bottom = torch.sqrt((torch.sum(torch.pow(y_d, 2), 0) * torch.sum(torch.pow(y_hat_d, 2), 0)))
                corr_inter = corr_top / corr_bottom
                val_corr[key] = ((1. / num_var) * torch.sum(corr_inter)).item()
                val_mae[key] = ((1. / (sample_num * num_var)) * torch.sum(torch.abs(y_diff))).item()

        elif self.hparams.powerset != 'all':
            key = self.hparams.powerset
            # take values and consider possible calendar data
            y_hat = y_hat_org[:,series['all'][0]:series['all'][1]]
            y = y_org[:,series['all'][0]:series['all'][1]]

            num_var = y.size(-1)
            sample_num = y.size(0)
            
            y_diff = y_hat - y
            y_mean = torch.mean(y)
            y_translation = y - y_mean
            
            val_maape[key] = (torch.mean(torch.atan(torch.abs((y-y_hat) / y)))).item()
            val_rrse[key] = (torch.sqrt(torch.sum(torch.pow(y_diff, 2))) / torch.sqrt(torch.sum(torch.pow(y_translation, 2)))).item()
            val_rse[key] = (torch.mean(torch.pow(y_diff, 2)) / torch.sum(torch.pow(y_translation, 2))).item()
            val_nrmse[key] = (torch.sqrt(torch.mean(torch.pow(y_diff, 2))) / torch.mean(torch.abs(y))).item()
            val_nd[key] = (torch.sum(torch.abs(y_diff)) / torch.sum(torch.abs(y))).item()

            # val_rho10', 'val_rho50', 'val_rho90'
            val_rho10[key] = (2 * (torch.sum((y - y_hat) * 0.1 * (y >= y_hat).float()) + torch.sum(
                (y_hat - y) * (1 - 0.1) * (y < y_hat).float())) / torch.sum(torch.abs(y))).item()
            val_rho50[key] = (2 * (torch.sum((y - y_hat) * 0.5 * (y >= y_hat).float()) + torch.sum(
                (y_hat - y) * (1 - 0.5) * (y < y_hat).float())) / torch.sum(torch.abs(y))).item()
            val_rho90[key] = (2 * (torch.sum((y - y_hat) * 0.9 * (y >= y_hat).float()) + torch.sum(
                (y_hat - y) * (1 - 0.9) * (y < y_hat).float())) / torch.sum(torch.abs(y))).item()

            y_m = torch.mean(y, 0, True)
            y_hat_m = torch.mean(y_hat, 0, True)
            y_d = y - y_m
            y_hat_d = y_hat - y_hat_m
            corr_top = torch.sum(y_d * y_hat_d, 0)
            corr_bottom = torch.sqrt((torch.sum(torch.pow(y_d, 2), 0) * torch.sum(torch.pow(y_hat_d, 2), 0)))
            corr_inter = corr_top / corr_bottom
            val_corr[key] = ((1. / num_var) * torch.sum(corr_inter)).item()
            val_mae[key] = ((1. / (sample_num * num_var)) * torch.sum(torch.abs(y_diff))).item()

        else:
            key = 'all'
            y_hat = y_hat_org[:,series[key][0]:series[key][1]]
            y = y_org[:,series[key][0]:series[key][1]]
            num_var = y.size(-1)
            sample_num = y.size(0)

            y_diff = y_hat - y
            y_mean = torch.mean(y)
            y_translation = y - y_mean

            val_maape[key] = (torch.mean(torch.atan(torch.abs((y-y_hat) / y)))).item()
            val_rrse[key] = (torch.sqrt(torch.sum(torch.pow(y_diff, 2))) / torch.sqrt(torch.sum(torch.pow(y_translation, 2)))).item()
            val_rse[key] = (torch.mean(torch.pow(y_diff, 2)) / torch.sum(torch.pow(y_translation, 2))).item()
            val_nrmse[key] = (torch.sqrt(torch.mean(torch.pow(y_diff, 2))) / torch.mean(torch.abs(y))).item()
            val_nd[key] = (torch.sum(torch.abs(y_diff)) / torch.sum(torch.abs(y))).item()

            # val_rho10', 'val_rho50', 'val_rho90'
            val_rho10[key] = (2 * (torch.sum((y - y_hat) * 0.1 * (y >= y_hat).float()) + torch.sum(
                (y_hat - y) * (1 - 0.1) * (y < y_hat).float())) / torch.sum(torch.abs(y))).item()
            val_rho50[key] = (2 * (torch.sum((y - y_hat) * 0.5 * (y >= y_hat).float()) + torch.sum(
                (y_hat - y) * (1 - 0.5) * (y < y_hat).float())) / torch.sum(torch.abs(y))).item()
            val_rho90[key] = (2 * (torch.sum((y - y_hat) * 0.9 * (y >= y_hat).float()) + torch.sum(
                (y_hat - y) * (1 - 0.9) * (y < y_hat).float())) / torch.sum(torch.abs(y))).item()

            y_m = torch.mean(y, 0, True)
            y_hat_m = torch.mean(y_hat, 0, True)
            y_d = y - y_m
            y_hat_d = y_hat - y_hat_m
            corr_top = torch.sum(y_d * y_hat_d, 0)
            corr_bottom = torch.sqrt((torch.sum(torch.pow(y_d, 2), 0) * torch.sum(torch.pow(y_hat_d, 2), 0)))
            corr_inter = corr_top / corr_bottom
            val_corr[key] = ((1. / num_var) * torch.sum(corr_inter)).item()
            val_mae[key] = ((1. / (sample_num * num_var)) * torch.sum(torch.abs(y_diff))).item()
        """
        if self.trainer.resume_from_checkpoint is not None:
            path_to_checkpoint = self.trainer.resume_from_checkpoint
            path_to_checkpoint = path_to_checkpoint.split('\\checkpoints')
        else: 
            path_to_checkpoint = self.trainer.model.trainer.model.trainer.logger._experiment.log_dir.split('\\tf')

        with open('{}\\{}'.format(path_to_checkpoint[0], 'yhat.csv'), 'ab') as f:
            #f.write(b'\n')
            np.savetxt(f, y_hat.cpu(), delimiter=",")
        """
        tqdm_dict ={'val_loss': val_loss_mean.cpu()}
        errormeasures=['nd', 'nrmse', 'maape', 'rrse', 'corr', 'mae', 'rse', 'rho10', 'rho50', 'rho90']
        for item in errormeasures:
            for key in val_nd:
                keyvalue='{}{}{}{}'.format('val_', item, '_',  key)
                toeval = '{}{}{}{}{}'.format('val_', item, '[\'',  key, '\']')
                tqdm_dict[keyvalue] = eval(toeval)


        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
        
        if self.val_results is None or tqdm_dict['val_loss'] < self.val_results['val_loss']:
            #if self.val_results is not None:
                #print(f"Best: {tqdm_dict['val_loss']} < {self.val_results['val_loss']}")
            #else:
                #print(f"Best: {tqdm_dict['val_loss']}")
            self.val_results = tqdm_dict
        #else:
        #    #print(f"Result: {tqdm_dict['val_loss']} >= {self.val_results['val_loss']}")

        return result



    def test_step(self, data, batch_idx):
        """
	    Lightning calls this inside the test loop
	    """
        #print(f'test_step: We are testing... {str(self.trainer.testing)}')
        x, y = data
        #x = x.view(x.size(0), -1)
        
        if self.hparams.mcdropout == 'True':
            self.hparams.mcdropout = 'False'
            y_hat = self.forward(x)
            loss_test = self.loss(y, y_hat)
            self.hparams.mcdropout = 'True'
            #print(f"Do the runs")
            y_hats = y_hat
            ys = y
            for loopID in range(100):
                y_hat = self.forward(x)
                y_hats = torch.cat(([y_hats, y_hat]), 0)
                ys = torch.cat(([ys, y]), 0)
            local_hparams = self.hparams
            if hasattr(local_hparams, 'scaler'):
                if self.on_gpu:
                    ys_inversed = torch.tensor(self.dataset.scaler.inverse_transform(torch.squeeze(ys).cpu())).cuda(x.device.index)
                    y_hats_inversed = torch.tensor(self.dataset.scaler.inverse_transform(torch.squeeze(y_hats.detach().cpu()))).cuda(x.device.index)
                else: 
                    ys_inversed = torch.tensor(self.scaler.inverse_transform(torch.squeeze(ys).cpu()))
                    y_hats_inversed = torch.tensor(self.scaler.inverse_transform(torch.squeeze(y_hats.detach().cpu())))

                ys_inversed = torch.unsqueeze(ys_inversed, 1)
                y_hats_inversed = torch.unsqueeze(y_hats_inversed, 1)
            elif hasattr(local_hparams, 'scale'):
                if self.on_gpu:
                    ys_inversed = torch.squeeze(ys) * torch.tensor(self.dataset.scale).cuda(x.device.index)
                    y_hats_inversed = torch.squeeze(y_hats) * torch.tensor(self.dataset.scale).cuda(x.device.index)
                else:
                    ys_inversed = torch.squeeze(ys) * torch.tensor(self.dataset.scale)
                    y_hats_inversed = torch.squeeze(y_hats) * torch.tensor(self.dataset.scale)

                ys_inversed = torch.unsqueeze(ys_inversed, 1)
                y_hats_inversed = torch.unsqueeze(y_hats_inversed, 1)
            else:
                ys_inversed = ys
                y_hats_inversed = y_hats
        else:
            y_hat = self.forward(x)
            loss_test = self.loss(y, y_hat)
            ys_inversed = y
            y_hats_inversed = y_hat

        local_hparams = self.hparams
        if hasattr(local_hparams, 'scaler'):
            if self.on_gpu:
                y_inversed = torch.tensor(self.dataset.scaler.inverse_transform(torch.squeeze(y).cpu())).cuda(x.device.index)
                y_hat_inversed = torch.tensor(self.dataset.scaler.inverse_transform(torch.squeeze(y_hat.detach().cpu()))).cuda(x.device.index)
            else: 
                y_inversed = torch.tensor(self.scaler.inverse_transform(torch.squeeze(y).cpu()))
                y_hat_inversed = torch.tensor(self.scaler.inverse_transform(torch.squeeze(y_hat.detach().cpu())))

            y_inversed = torch.unsqueeze(y_inversed, 1)
            y_hat_inversed = torch.unsqueeze(y_hat_inversed, 1)
        elif hasattr(local_hparams, 'scale'):
            if self.on_gpu:
                y_inversed = torch.squeeze(y) * torch.tensor(self.dataset.scale).cuda(x.device.index)
                y_hat_inversed = torch.squeeze(y_hat) * torch.tensor(self.dataset.scale).cuda(x.device.index)
            else:
                y_inversed = torch.squeeze(y) * torch.tensor(self.dataset.scale)
                y_hat_inversed = torch.squeeze(y_hat) * torch.tensor(self.dataset.scale)

            y_inversed = torch.unsqueeze(y_inversed, 1)
            y_hat_inversed = torch.unsqueeze(y_hat_inversed, 1)
        else:
            y_inversed = y
            y_hat_inversed = y_hat
   
        unscaled_loss_val = self.loss(y_inversed, y_hat_inversed)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_test = loss_test.unsqueeze(0)
            unscaled_loss_val = unscaled_loss_val.unsqueeze(0)

        output = OrderedDict({
            'val_loss': loss_test,
            'test_loss': loss_test,
            'y': y,
            'y_hat': y_hat,
            'y_inversed': y_inversed,
            'y_hat_inversed': y_hat_inversed,
            'unscaled_loss': unscaled_loss_val,
            'ys_inversed': ys_inversed,
            'y_hats_inversed': y_hats_inversed
        })

        # can also return just a scalar instead of a dict (return loss_test)
        return output

    def test_epoch_end(self, outputs):
        """
	    Called at the end of validation to aggregate outputs
		:param outputs: list of individual outputs of each test step
		"""
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()
        print('test_epoch_end: We are collecting test results..')
        test_loss_mean = 0
        for output in outputs:
            test_loss = output['test_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp:
                test_loss = torch.mean(test_loss)
            test_loss_mean += test_loss

        test_loss_mean /= len(outputs)

        y = torch.cat(([x['y_inversed'] for x in outputs]), 0)
        y_hat = torch.cat(([x['y_hat_inversed'] for x in outputs]), 0)

        ys = torch.cat(([x['ys_inversed'] for x in outputs]), 0)
        y_hats = torch.cat(([x['y_hats_inversed'] for x in outputs]), 0)

        num_var = y.size(-1)
        y = y.view(-1, num_var)
        y_hat = y_hat.view(-1, num_var)
        ys = ys.view(-1, num_var)
        y_hats = y_hats.view(-1, num_var)
        sample_num = y.size(0)
        
        y_hat_org= y_hat
        y_org = y
        y_hats_org= y_hats
        ys_org = ys
        test_rrse={}
        test_rse={}
        test_nrmse={}
        test_nd={}
        test_rho10={}
        test_rho50={}
        test_rho90={}
        test_corr={}
        test_mae={}
        test_maape={}

        if self.hparams.calendar == 'True' or self.hparams.n_multiv == 189 or self.hparams.n_multiv == 327:
            series = {'all':[0,-6], 'load':[0,59], 'price':[59,90], 'wind':[90, 147], 'solar':[147, 183]}
        else:
            series = {'all':[0,None], 'load':[0,59], 'price':[59,90], 'wind':[90, 147], 'solar':[147, 183]}

        if self.hparams.data_name == 'europe_power_system' and self.hparams.powerset == 'all':
            for key in series:
                y_hat = y_hat_org[:,series[key][0]:series[key][1]]
                y = y_org[:,series[key][0]:series[key][1]]
                y_hats = y_hats_org[:,series[key][0]:series[key][1]]
                ys = ys_org[:,series[key][0]:series[key][1]]
                num_var = y.size(-1)
                sample_num = y.size(0)

                y_diff = y_hat - y
                y_mean = torch.mean(y)
                y_translation = y - y_mean

                test_maape[key] = (torch.mean(torch.atan(torch.abs((y-y_hat) / y)))).item()
                test_rrse[key] = (torch.sqrt(torch.sum(torch.pow(y_diff, 2))) / torch.sqrt(torch.sum(torch.pow(y_translation, 2)))).item()
                test_rse[key] = (torch.mean(torch.pow(y_diff, 2)) / torch.sum(torch.pow(y_translation, 2))).item()
                test_nrmse[key] = (torch.sqrt(torch.mean(torch.pow(y_diff, 2))) / torch.mean(torch.abs(y))).item()
                test_nd[key] = (torch.sum(torch.abs(y_diff)) / torch.sum(torch.abs(y))).item()

                # val_rho10', 'val_rho50', 'val_rho90'
                test_rho10[key] = (2 * (torch.sum((ys - y_hats) * 0.1 * (ys >= y_hats).float()) + torch.sum(
                    (y_hats - ys) * (1 - 0.1) * (ys < y_hats).float())) / torch.sum(torch.abs(ys))).item()
                test_rho50[key] = (2 * (torch.sum((ys - y_hats) * 0.5 * (ys >= y_hats).float()) + torch.sum(
                    (y_hats - ys) * (1 - 0.5) * (ys < y_hats).float())) / torch.sum(torch.abs(ys))).item()
                test_rho90[key] = (2 * (torch.sum((ys - y_hats) * 0.9 * (ys >= y_hats).float()) + torch.sum(
                    (y_hats - ys) * (1 - 0.9) * (ys < y_hats).float())) / torch.sum(torch.abs(ys))).item()

                y_m = torch.mean(y, 0, True)
                y_hat_m = torch.mean(y_hat, 0, True)
                y_d = y - y_m
                y_hat_d = y_hat - y_hat_m
                corr_top = torch.sum(y_d * y_hat_d, 0)
                corr_bottom = torch.sqrt((torch.sum(torch.pow(y_d, 2), 0) * torch.sum(torch.pow(y_hat_d, 2), 0)))
                corr_inter = corr_top / corr_bottom
                test_corr[key] = ((1. / num_var) * torch.sum(corr_inter)).item()
                test_mae[key] = ((1. / (sample_num * num_var)) * torch.sum(torch.abs(y_diff))).item()

        elif self.hparams.powerset != 'all':
            key = 'all'
            # take values and consider possible calendar data
            y_hat = y_hat_org[:,series[key][0]:series[key][1]]
            y = y_org[:,series[key][0]:series[key][1]]
            y_hats = y_hats_org[:,series[key][0]:series[key][1]]
            ys = ys_org[:,series[key][0]:series[key][1]]

            key = self.hparams.powerset
            num_var = y.size(-1)
            sample_num = y.size(0)
            
            y_diff = y_hat - y
            y_mean = torch.mean(y)
            y_translation = y - y_mean
            
            test_maape[key] = (torch.mean(torch.atan(torch.abs((y-y_hat) / y)))).item()
            test_rrse[key] = (torch.sqrt(torch.sum(torch.pow(y_diff, 2))) / torch.sqrt(torch.sum(torch.pow(y_translation, 2)))).item()
            test_rse[key] = (torch.mean(torch.pow(y_diff, 2)) / torch.sum(torch.pow(y_translation, 2))).item()
            test_nrmse[key] = (torch.sqrt(torch.mean(torch.pow(y_diff, 2))) / torch.mean(torch.abs(y))).item()
            test_nd[key] = (torch.sum(torch.abs(y_diff)) / torch.sum(torch.abs(y))).item()

            # val_rho10', 'val_rho50', 'val_rho90'
            test_rho10[key] = (2 * (torch.sum((ys - y_hats) * 0.1 * (ys >= y_hats).float()) + torch.sum(
                (y_hats - ys) * (1 - 0.1) * (ys < y_hats).float())) / torch.sum(torch.abs(ys))).item()
            test_rho50[key] = (2 * (torch.sum((ys - y_hats) * 0.5 * (ys >= y_hats).float()) + torch.sum(
                (y_hats - ys) * (1 - 0.5) * (ys < y_hats).float())) / torch.sum(torch.abs(ys))).item()
            test_rho90[key] = (2 * (torch.sum((ys - y_hats) * 0.9 * (ys >= y_hats).float()) + torch.sum(
                (y_hats - ys) * (1 - 0.9) * (ys < y_hats).float())) / torch.sum(torch.abs(ys))).item()

            y_m = torch.mean(y, 0, True)
            y_hat_m = torch.mean(y_hat, 0, True)
            y_d = y - y_m
            y_hat_d = y_hat - y_hat_m
            corr_top = torch.sum(y_d * y_hat_d, 0)
            corr_bottom = torch.sqrt((torch.sum(torch.pow(y_d, 2), 0) * torch.sum(torch.pow(y_hat_d, 2), 0)))
            corr_inter = corr_top / corr_bottom
            test_corr[key] = ((1. / num_var) * torch.sum(corr_inter)).item() 
            test_mae[key] = ((1. / (sample_num * num_var)) * torch.sum(torch.abs(y_diff))).item()

        else:
            key = 'all'
            y_hat = y_hat_org[:,series[key][0]:series[key][1]]
            y = y_org[:,series[key][0]:series[key][1]]
            y_hats = y_hats_org[:,series[key][0]:series[key][1]]
            ys = ys_org[:,series[key][0]:series[key][1]]

            num_var = y.size(-1)
            sample_num = y.size(0)

            y_diff = y_hat - y
            y_mean = torch.mean(y)
            y_translation = y - y_mean

            test_maape[key] = (torch.mean(torch.atan(torch.abs((y-y_hat) / y)))).item()
            test_rrse[key] = (torch.sqrt(torch.sum(torch.pow(y_diff, 2))) / torch.sqrt(torch.sum(torch.pow(y_translation, 2)))).item()
            test_rse[key] = (torch.mean(torch.pow(y_diff, 2)) / torch.sum(torch.pow(y_translation, 2))).item()
            test_nrmse[key] = (torch.sqrt(torch.mean(torch.pow(y_diff, 2))) / torch.mean(torch.abs(y))).item()
            test_nd[key] = (torch.sum(torch.abs(y_diff)) / torch.sum(torch.abs(y))).item()

            # val_rho10', 'val_rho50', 'val_rho90'
            test_rho10[key] = (2 * (torch.sum((ys - y_hats) * 0.1 * (ys >= y_hats).float()) + torch.sum(
                (y_hats - ys) * (1 - 0.1) * (ys < y_hats).float())) / torch.sum(torch.abs(ys))).item()
            test_rho50[key] = (2 * (torch.sum((ys - y_hats) * 0.5 * (ys >= y_hats).float()) + torch.sum(
                (y_hats - ys) * (1 - 0.5) * (ys < y_hats).float())) / torch.sum(torch.abs(ys))).item()
            test_rho90[key] = (2 * (torch.sum((ys - y_hats) * 0.9 * (ys >= y_hats).float()) + torch.sum(
                (y_hats - ys) * (1 - 0.9) * (ys < y_hats).float())) / torch.sum(torch.abs(ys))).item()

            y_m = torch.mean(y, 0, True)
            y_hat_m = torch.mean(y_hat, 0, True)
            y_d = y - y_m
            y_hat_d = y_hat - y_hat_m
            corr_top = torch.sum(y_d * y_hat_d, 0)
            corr_bottom = torch.sqrt((torch.sum(torch.pow(y_d, 2), 0) * torch.sum(torch.pow(y_hat_d, 2), 0)))
            corr_inter = corr_top / corr_bottom
            test_corr[key] = ((1. / num_var) * torch.sum(corr_inter)).item()
            test_mae[key] = ((1. / (sample_num * num_var)) * torch.sum(torch.abs(y_diff))).item()

        if self.trainer.resume_from_checkpoint is not None:
            path_to_checkpoint = self.trainer.resume_from_checkpoint
            path_to_checkpoint = path_to_checkpoint.split('\\checkpoints')
        else: 
            path_to_checkpoint = self.trainer.model.trainer.model.trainer.logger._experiment.log_dir.split('\\tf')

        with open('{}\\{}'.format(path_to_checkpoint[0], 'y.csv'), 'ab') as f:
            #f.write(b'\n')
            np.savetxt(f, y_org.cpu(), delimiter=",")

        with open('{}\\{}'.format(path_to_checkpoint[0], 'yhat.csv'), 'ab') as f:
            #f.write(b'\n')
            np.savetxt(f, y_hat.cpu(), delimiter=",")

        with open('{}\\{}'.format(path_to_checkpoint[0], 'ys.csv'), 'ab') as f:
            #f.write(b'\n')
            np.savetxt(f, ys_org.cpu(), delimiter=",")

        with open('{}\\{}'.format(path_to_checkpoint[0], 'yhats.csv'), 'ab') as f:
            #f.write(b'\n')
            np.savetxt(f, y_hats_org.cpu(), delimiter=",")

        
        tqdm_dict ={'test_loss': test_loss_mean.cpu()}
        errormeasures=['nd', 'nrmse', 'maape', 'rrse', 'corr', 'mae', 'rse', 'rho10', 'rho50', 'rho90']
        for item in errormeasures:
            for key in test_nd:
                keyvalue='{}{}{}{}'.format('test_', item, '_',  key)
                toeval = '{}{}{}{}{}'.format('test_', item, '[\'',  key, '\']')
                tqdm_dict[keyvalue] = eval(toeval)


        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'test_loss': test_loss_mean}
        self.test_results = tqdm_dict
        return result

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
		return whatever optimizers we want here
		:return: list of optimizers
		"""
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]  # It is encouraged to try more optimizers and schedulers here

    def prepare_data(self):
        #  dataset, train, valid, horizon, window, normalise = 0):
        self.dataset = DataUtil(self.hparams, 2)

    def __dataloader(self, train):
        # init data generators
        if train == 'train':
            used_dataset = self.dataset.train
            should_shuffle = True
        elif train == 'validation':
            used_dataset = self.dataset.valid
            should_shuffle = False
        else:
            used_dataset = self.dataset.test
            should_shuffle = False

        batch_size = self.hparams.batch_size

        loader = DataLoader(TensorDataset(torch.Tensor(used_dataset[0]), torch.Tensor(used_dataset[1])),
            batch_size=batch_size,
            drop_last=True,
            shuffle=should_shuffle,
            #num_workers=4,
            #shuffle=should_shuffle,
            #sampler=train_sampler
        )

        return loader

    def train_dataloader(self):
        log.info('train data loader called')
        return self.__dataloader(train='train')

    def val_dataloader(self):
        log.info('val data loader called')
        return self.__dataloader(train='validation')

    def test_dataloader(self):
        log.info('test data loader called')
        return self.__dataloader(train='test')

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
		Parameters you define here will be available to your model through self.hparams
		"""
        parser = ArgumentParser(parents=[parent_parser])

        # param overwrites
        # parser.set_defaults(gradient_clip_val=5.0)

        # network params
        parser.add_argument('--local', default=3, type=int) # default 3
        parser.add_argument('--n_kernels', default=32, type=int)
        parser.add_argument('--w_kernel', type=int, default=1)
        parser.add_argument('--d_model', type=int, default=512)
        parser.add_argument('--d_inner', type=int, default=2048)
        parser.add_argument('--d_k', type=int, default=64)
        parser.add_argument('--d_v', type=int, default=64)
        parser.add_argument('--n_head', type=int, default=8)
        parser.add_argument('--n_layers', type=int, default=6)
        parser.add_argument('--drop_prob', type=float, default=0.1)

        def list_of_lists(arg):
            return [float(arg.split(','))]

        # arguments from dataset
        parser.add_argument('--data_name', default='europe_power_system', type=str)
        parser.add_argument('--data_dir', default='./data', type=str)
        parser.add_argument('--split_train', default=0.05, type=float)
        parser.add_argument('--split_validation', default=0.02, type=float)
        parser.add_argument('--split_test', default=0.001, type=float)
        parser.add_argument('--powerset', default='all', type=str)
        parser.add_argument('--calendar', default='False', type=str)
        parser.add_argument('--mcdropout', default='False', type=str)
        #{'all', 'load':[0,59], 'price':[59,90], 'wind':[90, 147], 'solar':[147, 183]}

        parser.add_argument('--n_multiv', default=1, type=int)
        parser.add_argument('--test_only', default=False, type=bool)

        parser.add_argument('--window', default=6, type=int)
        parser.add_argument('--horizon', default=3, type=int)

        # training params (opt)
        parser.add_argument('--learning_rate', default=0.005, type=float)
        parser.add_argument('--optimizer_name', default='adam', type=str)
        parser.add_argument('--criterion', default='mse_loss', type=str)

        # if using 2 nodes with 4 gpus each the batch size here (256) will be 256 / (2*8) = 16 per gpu
        parser.add_argument('--batch_size', default=16, type=int, help='batch size will be divided over all the gpus being used across all nodes')
        return parser
