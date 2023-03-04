# Load the Dataset and grab a subset
# In this python code we grab the dataset and scale a subset of dataset to use as
# sample workload in our application.

import logging
import re
import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch import xlogy
from datetime import timedelta
import math
from sklearn.metrics import mean_squared_error

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
sys.path.append(rootPath)

from data.handle_data_set2 import read_dataset_by_name, split, stationarity, get_resample_train_test_file_dir
from my_common.utils import get_logger
from data.handle_data_set2 import DATA_SET_NAME, RESAMPLE_FREQ_MIN, TEST_LEN, RESAMPLE_FREQ_MIN_STRING, TRAIN_LEN


def arima_stable_pdq(train, test, y_len, fre_min, p, d, q):
    # p=2
    # d=0
    # q=0
    arima_logger = get_logger('arima', logging.INFO)
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    history_df = train.copy(deep=True)
    predictions_series = pd.Series([], dtype='float64')
    resample_train_test_file_dir = get_resample_train_test_file_dir(
        DATA_SET_NAME, TEST_LEN, TRAIN_LEN, RESAMPLE_FREQ_MIN)
    index=-1*(y_len-1)
    while index <0:
        model = ARIMA(history_df[:index], order=(p, d, q)).fit()
        model.summary()
        forecast = model.forecast(y_len)
        predictions_series[forecast.index[1]] = forecast[1]
        index=index+1
    for t in range(len(test)):
        if (history_df.index[-1]+timedelta(minutes=2*fre_min) <= test.index[-1]):
            model = ARIMA(history_df, order=(p, d, q)).fit()
            model.summary()
            forecast = model.forecast(y_len)
            predictions_series[forecast.index[1]] = forecast[1]
            history_df.append(
                pd.Series({"count": test.iloc[t]}, name=test.index[t]))
            history_df.loc[test.index[t], 'count'] = test.iloc[t]['count']
            arima_logger.debug(
                'predict time: '+str(forecast.index[1])+' count: '+str(forecast[1]))
            arima_logger.debug(
                'test time: '+str(forecast.index[1])+' count: '+str(test.loc[forecast.index[1], 'count']))
    pdq_mse = mean_squared_error(
        test['count'].values, predictions_series.values)
    
    csv_save_path = os.path.join('prediction_result/' + resample_train_test_file_dir+'/arima')
    fig_save_path=os.path.join('prediction_result/'+ resample_train_test_file_dir+'/arima/fig')

    if not os.path.exists(csv_save_path):
        os.makedirs(csv_save_path)

    if not os.path.exists(fig_save_path):
        os.makedirs(fig_save_path)

    predictions_series.to_csv(csv_save_path+'/p_'+str(p)+'_d_'+str(d)+'_q_'+str(q)+'_prediction.csv')
    sns.set()
    fig = plt.figure(figsize=(15, 5))
    plt.plot(test, label='test')
    plt.plot(predictions_series, label='prediction')

    plt.legend()
    plt.savefig(fig_save_path+'/p_'+str(p)+'_d_'+str(d)+'_q_'+str(q)+'_prediction', dpi=400,
                bbox_inches='tight')  # transparent=True#
    return pdq_mse


def do_arima_predict(train_df, test_df, fre_min):
    # max_p = 10
    # max_q = 10
    # max_d = 3
    # for p in range(max_p+1):
    #     for q in range(max_q+1):
    #         for d in range(max_d+1):
    #             arima_stable_pdq(train_df, test_df, 2, fre_min, p, d, q)

    arima_stable_pdq(train_df, test_df, 2, fre_min, 2, 0, 0)


def find_best_pdq(test_df):
    path = '/home/wyz/time_series_prediction/data_use_stable_pdq/fre_40min/prediction_result/'
    best_pdq_csv_name = ''
    best_mse = 100
    for pdq_csv_name in os.listdir(path):
        prediction_df = pd.read_csv(
            path+pdq_csv_name, index_col=0, parse_dates=True)
        pdq_mse = mean_squared_error(
            test_df['count'].values[1:], prediction_df['0'].values)
        if pdq_mse < best_mse:
            best_mse = pdq_mse
            best_pdq_csv_name = pdq_csv_name
    print('best_pdq_csv_name'+best_pdq_csv_name)
    print('best_mse'+str(best_mse))


def main():
    # 读取测试数据集
    scaled_df = read_dataset_by_name(
        rootPath+'/data/'+DATA_SET_NAME, RESAMPLE_FREQ_MIN_STRING)
    # 测试平稳性
    stationarity(scaled_df)
    # 划分测试集和训练集，并保存
    train_df, test_df = split(scaled_df, test_len=TEST_LEN)
    do_arima_predict(train_df, test_df, RESAMPLE_FREQ_MIN)
    # find_best_pdq(test_df)


if __name__ == "__main__":
    main()
