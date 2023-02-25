# Load the Dataset and grab a subset
# In this python code we grab the dataset and scale a subset of dataset to use as
# sample workload in our application.
import logging
from my_common.utils import get_logger
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


def read_file_by_name(file_name, freq):
    df = pd.read_csv(file_name+'.csv', index_col=0, parse_dates=True)
    df = df.groupby('period').sum()
    # 放缩到最大流量为scaled_max
    scaled_max_normal = 100
    scaled_df = (df / df['count'].max() *
                 scaled_max_normal).apply(lambda x: round(x))
    scaled_df['count'] = scaled_df['count'].apply(lambda x: int(x))
    resample_df = scaled_df.resample(freq).mean()
    resample_df['count'] = resample_df['count'].astype(float)
    # print(resample_df)
    return resample_df


def draw_data_set(scaled_df, file_name):

    sns.set()
    fig = plt.figure(figsize=(15, 5))

    plt.xlabel('Time')
    # plt.plot(scaled_df['count'], label='Request(request/s)')
    plt.plot(np.array(list(range(len(scaled_df)))),
             scaled_df['count'].values, label='Request(request/s)')
    plt.legend(loc='upper right', fontsize=8)  # 标签位置
    plt.grid(color='gray')
    ax = plt.gca()  # 获取当前的axes
    ax.spines['right'].set_color('gray')
    ax.spines['left'].set_color('gray')
    ax.spines['top'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    # plt.xlim(-10, len(resample_df_low.index) + 10)
    # plt.ylim(0, 80)
    # plt.savefig('nasa_diff_threshold.png', dpi=400, bbox_inches='tight')
    plt.savefig(file_name+'.png', dpi=400, bbox_inches='tight')

# Testing For Stationarity


def stationarity(scaled_df):
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(scaled_df['count'])
    labels = ['ADF Test Statistic', 'p-value',
              '#Lags Used', 'Number of Observations Used']
    for value, label in zip(result, labels):
        print(label+' : '+str(value))
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")


# def arima(train, test, y_len, fre_min):
#     arima_logger = get_logger('arima', logging.INFO)
#     from statsmodels.tsa.arima.model import ARIMA
#     from statsmodels.tools.sm_exceptions import ConvergenceWarning
#     import warnings
#     history_df = train.copy(deep=True)
#     predictions_series = pd.Series([], dtype='float64')
#     for t in range(len(test)):
#         if (history_df.index[-1]+timedelta(minutes=2*fre_min) <= test.index[-1]):
#             pmax = int(len(history_df)/10)
#             qmax = int(len(history_df)/10)
#             bic_matrix = []
#             for p in range(pmax+1):
#                 tmp = []
#                 for q in range(qmax+1):
#                     try:
#                         tmp.append(
#                             ARIMA(history_df, order=(p, 1, q)).fit().bic)
#                     except:
#                         tmp.append(None)
#                 bic_matrix.append(tmp)
#             bic_matrix = pd.DataFrame(bic_matrix)
#             arima_logger.info(bic_matrix)
#             p, q = bic_matrix.stack().astype('float64').idxmin()
#             arima_logger.info(u'bic最小的P值和q值为：%s、%s' % (p, q))
#             model = ARIMA(history_df, order=(p, 1, q)).fit()
#             model.summary()
#             forecast = model.forecast(y_len)
#             predictions_series[forecast.index[1]] = forecast[1]
#             history_df.append(
#                 pd.Series({"count": test.iloc[t]}, name=test.index[t]))
#             history_df.loc[test.index[t], 'count'] = test.iloc[t]['count']
#             arima_logger.info(
#                 'predict time: '+str(forecast.index[1])+' count: '+str(forecast[1]))
#             arima_logger.info(
#                 'test time: '+str(forecast.index[1])+' count: '+str(test.loc[forecast.index[1], 'count']))
#     error = mean_squared_error(
#         test['count'].values[1:], predictions_series.values)
#     mean_squared_error_logger = get_logger('mean_squared_error', logging.INFO)
#     mean_squared_error_logger.info('mean_squared_error_logger: '+str(error))
#     predictions_series.to_csv('prediction.csv')
#     sns.set()
#     fig = plt.figure(figsize=(15, 5))
#     plt.plot(test.iloc[1:], label='test')
#     plt.plot(predictions_series, label='prediction')

#     plt.legend()
#     plt.savefig('prediction', dpi=400,
#                 bbox_inches='tight')  # transparent=True#


def arima_stable_pdq(train, test, y_len, fre_min, p, d, q):
    # p=2
    # d=0
    # q=0
    arima_logger = get_logger('arima', logging.INFO)
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    import warnings
    history_df = train.copy(deep=True)
    predictions_series = pd.Series([], dtype='float64')
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
        test['count'].values[1:], predictions_series.values)
    predictions_series.to_csv(
        'prediction_result/p_'+str(p)+'_d_'+str(d)+'_q_'+str(q)+'_prediction.csv')
    sns.set()
    fig = plt.figure(figsize=(15, 5))
    plt.plot(test.iloc[1:], label='test')
    plt.plot(predictions_series, label='prediction')

    plt.legend()
    plt.savefig('fig/p_'+str(p)+'_d_'+str(d)+'_q_'+str(q)+'_prediction', dpi=400,
                bbox_inches='tight')  # transparent=True#
    return pdq_mse


def split(scaled_df):
    test_len = 80
    test_df = scaled_df[-1*test_len:]
    train_df = scaled_df[:-1*test_len]

    # train_df=scaled_df[-1*(test_len+1000):-1*test_len]
    train_df.to_csv('train_df.csv')
    test_df.to_csv('test_df.csv')
    # easy test

    # scaled_df=scaled_df[0:100]
    # train_test_split=0.5
    # train_df=scaled_df[0:math.ceil(len(scaled_df)*train_test_split)]
    # test_df=scaled_df[math.ceil(len(scaled_df)*train_test_split):]

    # easy test
    return train_df, test_df


def do_arima_predict():
    max_p = 10
    max_q = 10
    max_d = 3
    for p in range(max_p+1):
        for q in range(max_q+1):
            for d in range(max_d+1):
                arima_stable_pdq(train_df, test_df, 2, fre_min, p, d, q)

    # arima_stable_pdq(train_df, test_df, 2, fre_min, 2, 0, 4)


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


# file_name='inttraffic'
file_name = 'inttraffic'
fre_min = 40
fre_min_string = str(fre_min)+'min'
scaled_df = read_file_by_name(file_name, fre_min_string)
draw_data_set(scaled_df, file_name)
stationarity(scaled_df)
train_df, test_df = split(scaled_df)
draw_data_set(test_df, file_name+'_test')

do_arima_predict()


find_best_pdq(test_df)
