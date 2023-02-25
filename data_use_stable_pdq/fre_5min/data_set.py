# Load the Dataset and grab a subset
# In this python code we grab the dataset and scale a subset of dataset to use as
# sample workload in our application.
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

from my_common.utils import get_logger
import logging

def read_file_by_name(file_name,freq):
    df = pd.read_csv(file_name+'.csv' , index_col=0, parse_dates=True)
    df = df.groupby('period').sum()
    # 放缩到最大流量为scaled_max
    scaled_max_normal = 100
    scaled_df = (df / df['count'].max() * scaled_max_normal).apply(lambda x: round(x))
    scaled_df['count'] = scaled_df['count'].apply(lambda x: int(x))
    resample_df = scaled_df.resample(freq).mean()
    resample_df['count']=resample_df['count'].astype(float)
    # print(resample_df)
    return resample_df


def draw_data_set(scaled_df,file_name):

    sns.set()
    fig = plt.figure(figsize=(15,5))

    plt.xlabel('Time')
    # plt.plot(scaled_df['count'], label='Request(request/s)')
    plt.plot(np.array(list(range(len(scaled_df)))),
         scaled_df['count'].values,label='Request(request/s)')
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

### Testing For Stationarity
def stationarity(scaled_df):
    from statsmodels.tsa.stattools import adfuller
    result=adfuller(scaled_df['count'])
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")


def arima(train,test,y_len):
    arima_logger=get_logger('arima',logging.INFO)
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    import warnings
    history_df=train.copy(deep=True)
    predictions_series=pd.Series([], dtype='float64')
    for t in range(len(test)):
        if(history_df.index[-1]+timedelta(minutes=2*20)<=test.index[-1]):
            pmax=int(len(history_df)/10)
            qmax=int(len(history_df)/10)
            bic_matrix=[]
            for p in range(pmax+1):
                tmp=[]
                for q in range(qmax+1):
                    try:
                        tmp.append(ARIMA(history_df, order=(p, 1, q)).fit().bic)
                    except:
                        tmp.append(None)
                bic_matrix.append(tmp)
            bic_matrix=pd.DataFrame(bic_matrix)
            arima_logger.info(bic_matrix)
            p,q=bic_matrix.stack().astype('float64').idxmin()
            arima_logger.info(u'bic最小的P值和q值为：%s、%s'%(p,q))
            model=ARIMA(history_df,order=(p,1,q)).fit()
            model.summary()
            forecast=model.forecast(y_len)
            predictions_series[forecast.index[1]]=forecast[1]
            history_df.append(pd.Series({"count":test.iloc[t]},name=test.index[t]))
            history_df.loc[test.index[t],'count']=test.iloc[t]['count']
            arima_logger.info('predict time: '+str(forecast.index[1])+' count: '+str(forecast[1]))
            arima_logger.info('test time: '+str(forecast.index[1])+' count: '+str(test.loc[forecast.index[1],'count']))
    error = mean_squared_error(test['count'].values[1:], predictions_series.values)
    mean_squared_error_logger=get_logger('mean_squared_error',logging.INFO)
    mean_squared_error_logger.info('mean_squared_error_logger: '+str(error))
    predictions_series.to_csv('prediction.csv')
    sns.set()
    fig = plt.figure(figsize=(15,5))
    plt.plot(test.iloc[1:],label='test')
    plt.plot(predictions_series,label='prediction')

    plt.legend()
    plt.savefig('prediction', dpi=400,
                bbox_inches='tight')  # transparent=True#


def split(scaled_df):
    test_len=650
    test_df=scaled_df[-1*test_len:]
    train_df=scaled_df[:-1*test_len]

    # train_df=scaled_df[-1*(test_len+1000):-1*test_len]
    train_df.to_csv('train_df.csv')
    test_df.to_csv('test_df.csv')
    # easy test

    # scaled_df=scaled_df[0:100]
    # train_test_split=0.5
    # train_df=scaled_df[0:math.ceil(len(scaled_df)*train_test_split)]
    # test_df=scaled_df[math.ceil(len(scaled_df)*train_test_split):]

    # easy test
    return train_df,test_df

# file_name='inttraffic'
file_name='inttraffic'
freq='5min'
scaled_df=read_file_by_name(file_name,freq)
draw_data_set(scaled_df,file_name)
stationarity(scaled_df)
train_df,test_df=split(scaled_df)
draw_data_set(test_df,file_name+'_test')

# arima(train_df,test_df,y_len=2)