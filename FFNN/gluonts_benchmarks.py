# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
This example shows how to fit a model and evaluate its predictions.
"""
import pprint
from functools import partial
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.append('/scratch/project_2002244/benchmarks/packages') #

# %matplotlib inline
import mxnet as mx
from mxnet import gluon

import matplotlib.pyplot as plt
import json
import os
from tqdm.autonotebook import tqdm
from pathlib import Path


from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.deepar import DeepAREstimator
# from gluonts.model.seq2seq import MQCNNEstimator
from gluonts.model.canonical import CanonicalRNNEstimator
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.gp_forecaster import GaussianProcessEstimator
from gluonts.model.lstnet import LSTNetEstimator
from gluonts.distribution.gaussian import GaussianOutput
from gluonts.distribution.student_t import StudentTOutput
from gluonts.trainer import Trainer
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.model.forecast import Config, OutputType

mx.random.seed(0)
np.random.seed(0)

def plot_prob_forecasts(ts_entry, forecast_entry, sample_id, prediction_length, plot_length, inline=True):
    prediction_intervals = (50, 67, 95, 99)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
    _, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    ax.axvline(ts_entry.index[-prediction_length], color='r')
    plt.legend(legend, loc="upper left")
    if inline:
        plt.show()
        plt.clf()


def get_custom_dataset(name, horizon):
    """
    """
    if name=="electricity":
        csv_path = r'/scratch/project_2002244/DeepAR/data/elect/electricity.csv'
        df = pd.read_csv(csv_path, sep=",", index_col=0, parse_dates=True, decimal='.').astype(float)
        df.fillna(0, inplace=True)
        train_start = '2012-01-01 00:00:00'
        train_end = '2014-05-26 23:00:00'
        test_start = '2014-05-27 00:00:00'
        test_end = '2014-12-31 23:00:00'
    elif name=="europe_power_system":
        csv_path = r'/scratch/project_2002244/DeepAR/data/elect/europe_power_system.csv'
        df = pd.read_csv(csv_path, sep=",", index_col=0, parse_dates=True, decimal='.').astype(float)
        df.fillna(0, inplace=True)
        train_start = '2015-01-01 00:00:00'
        train_end = '2017-06-23 23:00:00'
        test_start = '2017-06-24 00:00:00'
        test_end = '2017-11-30 23:00:00'

    train_target_values = df[:train_end].T.values
    test_target_values = df[:(pd.Timestamp(test_start)-timedelta(hours=1))].T.values
    start_dates = np.array([pd.Timestamp(df.index[0], freq='1H') for _ in range(train_target_values.shape[0])])

    train_ds = ListDataset([
            {
                FieldName.TARGET: target,
                FieldName.START: start
            }
            for (target, start) in zip(train_target_values, start_dates)
        ], freq="1H")

    test_ds = ListDataset([
            {
                FieldName.TARGET: target,
                FieldName.START: start
            }
            for index in pd.date_range(start=(pd.Timestamp(test_start)-timedelta(hours=1)+timedelta(hours=horizon)),
                                       end=pd.Timestamp(test_end), freq='{}H'.format(horizon))
                for (target, start) in zip(df[:index].T.values , start_dates)
        ], freq="1H")
    return train_ds, test_ds


datasets = [
    "electricity",
    "europe_power_system"
]

plot = False
save = True
ctx = mx.Context("cpu") #"gpu"
n_samples = 100
epochs = 500
num_batches_per_epoch = 50
learning_rate = 1e-3
freq = "H"
context_length = 168
batch_size = 64
horizons = [3, 6, 12, 24, 36]
patience = 25

estimators = [
    partial(
        SimpleFeedForwardEstimator,
        distr_output=GaussianOutput(),
        trainer=Trainer(
            ctx=ctx, epochs=epochs, num_batches_per_epoch=num_batches_per_epoch,
            batch_size=batch_size, learning_rate=learning_rate, patience=patience
        ),
    ),
    partial(
        DeepAREstimator,
        #distr_output = GaussianOutput(),
#         use_feat_static_cat=True,
#         cardinality=1,
        num_layers=3,
        trainer=Trainer(
            ctx=ctx, epochs=epochs, num_batches_per_epoch=num_batches_per_epoch,
            batch_size=batch_size, learning_rate=learning_rate, patience=patience
        ),
    ),
    partial(
        LSTNetEstimator,
        skip_size=24,
        channels=24*6,
        ar_window = 24,
        num_series=321,
        trainer=Trainer(
            ctx=ctx, epochs=epochs, num_batches_per_epoch=num_batches_per_epoch,
            batch_size=batch_size, learning_rate=learning_rate, patience=patience
        ),
    ),
    partial(
        CanonicalRNNEstimator,
        #distr_output=GaussianOutput(), #StudentTOutput(),
        #num_layers=2,
        #cell_type="lstm",
        #num_cells=100,
        cardinality = [1],
        trainer=Trainer(
            ctx=ctx, epochs=epochs, num_batches_per_epoch=num_batches_per_epoch,
            batch_size=batch_size, learning_rate=learning_rate, patience=patience
        ),
    ),
    partial(
        GaussianProcessEstimator,
#         cardinality = 1,
        trainer=Trainer(
            ctx=ctx, epochs=epochs, num_batches_per_epoch=num_batches_per_epoch,
            batch_size=batch_size, learning_rate=learning_rate, patience=patience
        ),
    ),
]


def evaluate(dataset_name, estimator, horizon):
    train_ds, test_ds = get_custom_dataset(dataset_name, horizon)
    estimator = estimator(
        prediction_length=horizon,
        freq=freq,
        context_length = context_length,
        #cardinality=len(train_ds)
    )

    print(f"evaluating {estimator} on {dataset_name} dataset for {horizon} horizon")

    predictor = estimator.train(train_ds)

    forecast_it, ts_it = make_evaluation_predictions(
        test_ds, predictor=predictor, num_samples=n_samples
    )

    print("Obtaining time series conditioning values ...")
    tss = list(tqdm(ts_it, total=len(test_ds)))
    print("Obtaining time series predictions ...")
    forecasts = list(tqdm(forecast_it, total=len(test_ds)))

    if plot:
        print("Plotting time series predictions ...")
        for i in tqdm(range(0, 361, 90)):
            ts_entry = tss[i]
            forecast_entry = forecasts[i]
            plot_prob_forecasts(ts_entry, forecast_entry, i, horizon, context_length)

    print("Saving time series predictions ...")
    series = int(len(forecasts)/len(train_ds))
    sesies_q = np.empty((0,horizon*series), float)
    q10_, q50_, q90_, indexes_ = sesies_q, sesies_q, sesies_q, np.empty((0,horizon*series),'datetime64[s]')
    for i in range(len(train_ds)):
        q10, q50, q90, indexes = np.array([]), np.array([]), np.array([]), np.array([])
        for z in range(series):
            f_dict = forecasts[z*len(train_ds)+i].as_json_dict(Config(output_types={OutputType.quantiles}))['quantiles']
            q10 = np.append(q10, np.array(f_dict['0.1']))
            q50 = np.append(q50, np.array(f_dict['0.5']))
            q90 = np.append(q90, np.array(f_dict['0.9']))
            indexes = np.append(indexes, np.array(list(forecasts[z*len(train_ds)+i].index)))
        q10_ = np.vstack((q10_, q10))
        q50_ = np.vstack((q50_, q50))
        q90_ = np.vstack((q90_, q90))
        indexes_ = np.vstack((indexes_, indexes))

    if save:
        save_file = r"./save/{}_{}_{}".format(type(estimator).__name__, dataset_name, str(horizon))
        np.savetxt('{}_q10.txt'.format(save_file), q10_)
        np.savetxt('{}_q50.txt'.format(save_file), q50_)
        np.savetxt('{}_q90.txt'.format(save_file), q90_)
        np.savetxt('{}_index.txt'.format(save_file), indexes_, fmt='%s')


    print("Calculating time series prediction metrics ...")
    agg_metrics, item_metrics = Evaluator()(
        iter(tss), iter(forecasts), num_series=len(test_ds)
    )

    pprint.pprint(agg_metrics)

    eval_dict = agg_metrics
    eval_dict["dataset"] = dataset_name
    eval_dict["estimator"] = type(estimator).__name__
    eval_dict["horizon"] = str(horizon)
    return eval_dict


if __name__ == "__main__":
    import gluonts
    print(gluonts.__version__)
    print(mx.__version__)

    results = []
    for horizon in horizons:
        for dataset_name in datasets:
            for estimator in estimators:
                # catch exceptions that are happening during training to avoid failing the whole evaluation
                #try:
                evals = evaluate(dataset_name, estimator, horizon)
                results.append(evals)
                #except Exception as e:
                #    print(str(e))

                df = pd.DataFrame(results)

                sub_df = df[
                    [
                        "dataset",
                        "estimator",
                        "horizon",
                        "ND",
                        "NRMSE",
                        "wQuantileLoss[0.1]",
                        "wQuantileLoss[0.5]",
                        "wQuantileLoss[0.9]",
                        "Coverage[0.1]",
                        "Coverage[0.5]",
                        "Coverage[0.9]",
                    ]
                ]

                print(sub_df.to_string())
                if save:
                    sub_df.to_csv(r"./save/metrics_benchmarks_deepar_el.csv",index=False)
