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


DATA_SET_NAME = 'WorldCup'
# RESAMPLE_FREQ_MIN = 40
# RESAMPLE_FREQ_MIN_STRING = str(RESAMPLE_FREQ_MIN)+'min'
# TEST_LEN = 80
# TRAIN_LEN = 1767

RESAMPLE_FREQ_MIN = 5
RESAMPLE_FREQ_MIN_STRING = str(RESAMPLE_FREQ_MIN)+'min'
TEST_LEN = 50
TRAIN_LEN = 100

def read_dataset_by_name(dataset_name, freq):
    df = pd.read_csv(dataset_name+'.csv', index_col=0, parse_dates=True)
    df = df.groupby('period').sum()
    df=df['1998-06-10 08:00:01':'1998-07-10 08:00:01']
    # 放缩到最大流量为scaled_max
    scaled_max_normal = 100
    scaled_df = (df / df['count'].max() *
                 scaled_max_normal).apply(lambda x: round(x))
    scaled_df['count'] = scaled_df['count'].apply(lambda x: int(x))
    resample_df = scaled_df.resample(freq).mean()
    resample_df['count'] = resample_df['count'].astype(float)
    # print(resample_df)
    return resample_df


def draw_data_set(scaled_df, dataset_name,file_name ):

    sns.set()
    fig = plt.figure(figsize=(15, 5))

    plt.xlabel('Time')
    plt.plot(scaled_df['count'], label='Request(request/s)')
    # plt.plot(np.array(list(range(len(scaled_df)))),
    #          scaled_df['count'].values, label='Request(request/s)')
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

    save_path = os.path.join(file_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plt.savefig(save_path+'/'+dataset_name+'.png', dpi=400, bbox_inches='tight')

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


def split(scaled_df, test_len):
    test_df = scaled_df[-1*test_len:]
    train_df = scaled_df[-1*(TRAIN_LEN+test_len):-1*test_len]

    # easy test

    # scaled_df=scaled_df[0:100]
    # train_test_split=0.5
    # train_df=scaled_df[0:math.ceil(len(scaled_df)*train_test_split)]
    # test_df=scaled_df[math.ceil(len(scaled_df)*train_test_split):]

    # easy test
    return train_df, test_df

# 读取测试数据集
def get_test_series(resample_train_test_file):
    test_data_file = rootPath+'/data/'+resample_train_test_file+'/test_df.csv'
    test_data_df = pd.read_csv(test_data_file, index_col=0, parse_dates=True)
    test_data_series = pd.Series(
        test_data_df['count'].values, index=test_data_df.index)
    return test_data_series

def get_train_series(resample_train_test_file):
    train_data_file = rootPath+'/data/'+resample_train_test_file+'/train_df.csv'
    train_data_df = pd.read_csv(train_data_file, index_col=0, parse_dates=True)
    train_data_series = pd.Series(
        train_data_df['count'].values, index=train_data_df.index)
    return train_data_series

def get_resample_train_test_file_dir(data_set_name, test_len, train_len, resample_interval_minite):
    resample_train_test_file_dir = data_set_name+'/resample_freq_'+str(resample_interval_minite) + \
        'min_test_len_'+str(test_len)+'_train_len_'+str(train_len)
    return resample_train_test_file_dir

def main():
    scaled_df = read_dataset_by_name(DATA_SET_NAME, RESAMPLE_FREQ_MIN_STRING)
    draw_data_set(scaled_df, DATA_SET_NAME, file_name=DATA_SET_NAME)
    # 测试平稳性
    stationarity(scaled_df)
    # 划分测试集和训练集，并保存
    train_df, test_df = split(scaled_df, test_len=TEST_LEN)
    path = os.path.join(DATA_SET_NAME+'/resample_freq_'+RESAMPLE_FREQ_MIN_STRING +
                        '_test_len_'+str(TEST_LEN)+'_train_len_'+str(TRAIN_LEN))
    if not os.path.exists(path):
        os.mkdir(path)
    train_df.to_csv(path+'/train_df.csv')
    test_df.to_csv(path+'/test_df.csv')
    
    draw_data_set(test_df, '/resample_freq_'+RESAMPLE_FREQ_MIN_STRING +
                        '_test_len_'+str(TEST_LEN)+'_train_len_'+str(TRAIN_LEN)+'/test',file_name=DATA_SET_NAME)

if __name__== "__main__" :
    main()