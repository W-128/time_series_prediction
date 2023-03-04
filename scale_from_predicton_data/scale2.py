
# 预测法
# 获取 train test prediction

# 静态阈值法
# 获取test

# 测试的test集肯定是一致的


import sys
import os
from datetime import timedelta
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
sys.path.append(rootPath)

from data.handle_data_set2 import RESAMPLE_FREQ_MIN, TEST_LEN, TRAIN_LEN, DATA_SET_NAME
from data.handle_data_set2 import get_test_series, get_resample_train_test_file_dir
pdq_prediction_csv = 'p_1_d_1_q_1_prediction.csv'

lstm_prediction_csv = 'ep1102-loss2.877-val_loss3.416_prediction.csv'

# 模拟RESMAPLE_INTERVAL和POD_INITIAL_SECONDS是一致的
POD_INITIAL_MINITE = RESAMPLE_FREQ_MIN

POD_REQ_THRESHOLD = 10.0
POD_CPU_UP_THRESHOLD = 0.8
POD_CPU_DOWN_THRESHOLD = 0.5
NORMAL_RESPONSE_TIME = 30
MORE_THAN_THRESHOLD_RESPONSE_TIME = 20000
POD_FIRST_NUM = 3
PREDICT_Y_LEN = 2
SLA = 1000
COLD_MINITE = RESAMPLE_FREQ_MIN*10


def static_threshold_method(test_data_series, pod_req_threshold,
                            pod_cpu_up_threshold, pod_cpu_down_threshold,
                            normal_reqsponse_time, more_than_threshold_response_time,
                            resample_interval_minute, pod_intial_minute, pod_first_num):
    pod_num_dic = {}
    pod_num_dic[test_data_series.index[0]] = pod_first_num
    response_time_dic = {}
    last_reduce_time = None
    for i, v in test_data_series.items():
        # 下一个时间点==可缩的时间点
        next_time_step = i+timedelta(minutes=resample_interval_minute)
        # 可扩的时间点
        next_can_add_pod_time_step = i + \
            timedelta(minutes=resample_interval_minute+pod_intial_minute)
        # 计算响应时间
        pod_num = pod_num_dic[i]
        if pod_num*pod_req_threshold*pod_cpu_up_threshold >= v:
            response_time_dic[i] = normal_reqsponse_time
        else:
            response_time_dic[i] = more_than_threshold_response_time

        if next_time_step > test_data_series.index[-1]:
            break

        # 下一个时间点的pod_num未被计算得出
        if next_time_step not in pod_num_dic:
            pod_num_dic[next_time_step] = pod_num

        pod_utilization = get_pod_utilization(
            req_num=v, p_num=pod_num, pod_req_threshold=pod_req_threshold)
        if pod_utilization > pod_cpu_up_threshold or pod_utilization < pod_cpu_down_threshold:
            need_pod = caculate_need_pod(
                p_utilization=pod_utilization, p_num=pod_num, pod_cpu_up_threshold=pod_cpu_up_threshold)
            # 扩
            if pod_utilization > pod_cpu_up_threshold:
                pod_num_dic[next_can_add_pod_time_step] = math.ceil(need_pod)
                last_reduce_time = next_can_add_pod_time_step
            # 缩
            else:
                # 冷静期已过
                if last_reduce_time == None or next_time_step >= last_reduce_time+timedelta(minutes=COLD_MINITE):
                    pod_num_dic[next_time_step] = math.ceil(need_pod)
                    pod_num_dic[next_can_add_pod_time_step] = math.ceil(
                        need_pod)
                    last_reduce_time = next_time_step

        # else:
        #     pod_num_dic[i+timedelta(seconds=2*resample_interval)
        #                 ] = pod_num_dic[i+timedelta(seconds=resample_interval)]

    return pod_num_dic, response_time_dic


def predict(test_data_series, prediction_series, pod_req_threshold,
            pod_cpu_up_threshold, pod_cpu_down_threshold,
            normal_reqsponse_time, more_than_threshold_response_time,
            resample_interval_minute, pod_intial_minute, pod_first_num
            ):

    pod_num_dic = {}
    response_time_dic = {}
    pod_num_dic[test_data_series.index[0]] = pod_first_num
    # for i in range(PREDICT_X_LEN):
    #     pod_num_dic[req_series.index[i]] = pod_first_num

    last_reduce_time = None
    for i, v in test_data_series.items():
        # 计算响应时间
        now_pod_num = pod_num_dic[i]
        if now_pod_num*pod_req_threshold*pod_cpu_up_threshold >= v:
            response_time_dic[i] = normal_reqsponse_time
        else:
            response_time_dic[i] = more_than_threshold_response_time

        # 下一个时间点==可缩的时间点
        next_time_step = i+timedelta(minutes=resample_interval_minute)
        # 可扩的时间点
        next_can_add_pod_time_step = i + \
            timedelta(minutes=resample_interval_minute+pod_intial_minute)

        req_num_hat = predict_method(
            i, resample_interval_minutes=resample_interval_minute, predict_y_len=PREDICT_Y_LEN, prediction_series=prediction_series)
        for k in range(len(req_num_hat)):
            predict_req_num = req_num_hat[k]
            need_pod = math.ceil(get_pod_num_according_req_num(req_num=predict_req_num,
                                                               pod_cpu_up_threshold=pod_cpu_up_threshold,
                                                               pod_req_threshold=pod_req_threshold))
            if need_pod < now_pod_num:
                # 冷静期已过
                if last_reduce_time == None or i+timedelta(minutes=(k+1)*resample_interval_minute) >= last_reduce_time+timedelta(minutes=COLD_MINITE):
                    pod_num_dic[i+timedelta(minutes=(k+1)
                                            * resample_interval_minute)] = need_pod
                    last_reduce_time = i + \
                        timedelta(minutes=(k+1)*resample_interval_minute)
            if need_pod > now_pod_num and i+timedelta(minutes=(k+1)*resample_interval_minute) > i+timedelta(minutes=pod_intial_minute):
                pod_num_dic[i+timedelta(minutes=(k+1)
                                        * resample_interval_minute)] = need_pod
                last_reduce_time = i + \
                    timedelta(minutes=(k+1) * resample_interval_minute)

        # 未进行扩缩
        if i+timedelta(minutes=resample_interval_minute) not in pod_num_dic:
            pod_num_dic[i +
                        timedelta(minutes=resample_interval_minute)] = now_pod_num

    return pod_num_dic, response_time_dic


def predict_with_slow_reduce(test_data_series, prediction_series, pod_req_threshold,
                             pod_cpu_up_threshold, pod_cpu_down_threshold,
                             normal_reqsponse_time, more_than_threshold_response_time,
                             resample_interval_minute, pod_intial_minute, pod_first_num
                             ):
    # 计算need_pod的cpu利用率上限
    pod_cpu_up_threshold_to_need_pod = pod_cpu_up_threshold-0.04
    pod_num_dic = {}
    response_time_dic = {}
    pod_num_dic[test_data_series.index[0]] = pod_first_num
    # for i in range(PREDICT_X_LEN):
    #     pod_num_dic[req_series.index[i]] = pod_first_num
    last_reduce_time = None
    for i, v in test_data_series.items():
        # 计算响应时间
        now_pod_num = pod_num_dic[i]
        if now_pod_num*pod_req_threshold*pod_cpu_up_threshold >= v:
            response_time_dic[i] = normal_reqsponse_time
        else:
            response_time_dic[i] = more_than_threshold_response_time

        # 下一个时间点==可缩的时间点
        next_time_step = i+timedelta(minutes=resample_interval_minute)
        # 可扩的时间点
        next_can_add_pod_time_step = i + \
            timedelta(minutes=resample_interval_minute+pod_intial_minute)

        req_num_hat = predict_method(
            i, resample_interval_minutes=resample_interval_minute, predict_y_len=PREDICT_Y_LEN, prediction_series=prediction_series)
        need_pod_list = []
        for k in range(len(req_num_hat)):
            predict_req_num = req_num_hat[k]
            need_pod = math.ceil(get_pod_num_according_req_num(req_num=predict_req_num,
                                                               pod_cpu_up_threshold=pod_cpu_up_threshold_to_need_pod,
                                                               pod_req_threshold=pod_req_threshold))
            need_pod_list.append(need_pod)

        # 是否都需要缩减
        all_reduce_pod_flag = True
        for j in range(len(need_pod_list)):
            if need_pod_list[j] >= now_pod_num:
                all_reduce_pod_flag = False

        # 进行扩缩
        for k in range(len(need_pod_list)):
            need_pod = need_pod_list[k]
            if need_pod < now_pod_num and all_reduce_pod_flag:
                # 冷静期已过
                if last_reduce_time == None or i+timedelta(minutes=(k+1) * resample_interval_minute) >= last_reduce_time+timedelta(minutes=COLD_MINITE):
                    pod_num_dic[i+timedelta(minutes=(k+1)
                                            * resample_interval_minute)] = need_pod
                    last_reduce_time = i + \
                        timedelta(minutes=(k+1) * resample_interval_minute)
            if need_pod > now_pod_num and i+timedelta(minutes=(k+1)*resample_interval_minute) > i+timedelta(minutes=pod_intial_minute):
                pod_num_dic[i+timedelta(minutes=(k+1)
                                        * resample_interval_minute)] = need_pod
                last_reduce_time = i + \
                    timedelta(minutes=(k+1) * resample_interval_minute)

        # 未进行扩缩
        if i+timedelta(minutes=resample_interval_minute) not in pod_num_dic:
            pod_num_dic[i +
                        timedelta(minutes=resample_interval_minute)] = now_pod_num
    # print(pod_num_dic.values())
    # print(response_time_dic.values())
    return pod_num_dic, response_time_dic


# 返回长度为从x_dt往后 predict_y_len长度的req_num列表
def predict_method(x_dt, resample_interval_minutes, predict_y_len, prediction_series):
    predict_y = []
    for i in range(predict_y_len):
        if x_dt+timedelta(minutes=(i+1)*resample_interval_minutes) in prediction_series.index:
            predict_y.append(
                prediction_series[x_dt+timedelta(minutes=(i+1)*resample_interval_minutes)])

    return predict_y


def get_pod_utilization(req_num, p_num, pod_req_threshold):
    return req_num/(p_num*pod_req_threshold)


def caculate_need_pod(p_utilization, p_num, pod_cpu_up_threshold):
    return (p_utilization*p_num)/pod_cpu_up_threshold


def get_pod_num_according_req_num(req_num, pod_cpu_up_threshold, pod_req_threshold):
    return req_num/(pod_req_threshold*pod_cpu_up_threshold)

# 计算各项指标


def caculate_metric(pod_num_dic, response_time_dic):
    sla_vio_time = 0.0
    for response_time in response_time_dic.values():
        if response_time > SLA:
            sla_vio_time = sla_vio_time+1
    sla_vio_rate = sla_vio_time/len(response_time_dic)

    print('平均pod_num:'+str(sum(list(pod_num_dic.values())) /
                           len(list(pod_num_dic.values()))))
    print('平均response_time:'+str(sum(response_time_dic.values()) /
                                 len(response_time_dic.values())))

    print('sla违约率:{:.1f}%'.format(100.0 * sla_vio_rate))


# 读取arima的预测数据集
def get_airima_predict_series(data_set_name, test_len, train_len, resample_interval_minite, pdq_prediction_csv):
    resample_train_test_file = data_set_name+'/resample_freq_'+str(resample_interval_minite) + \
        'min_test_len_'+str(test_len)+'_train_len_'+str(train_len)
    arima_prediction_series_file = rootPath + \
        '/prediction/prediction_result/'+resample_train_test_file + \
        '/arima/'+pdq_prediction_csv
    arima_prediction_df = pd.read_csv(
        arima_prediction_series_file, index_col=0, parse_dates=True)
    arima_prediction_series = pd.Series(
        arima_prediction_df['0'].values, index=arima_prediction_df.index)

    return arima_prediction_series

# 读取arima的预测数据集


def get_lstm_predict_series(data_set_name, test_len, train_len, resample_interval_minite, lstm_prediction_csv):
    resample_train_test_file = data_set_name+'/resample_freq_'+str(resample_interval_minite) + \
        'min_test_len_'+str(test_len)+'_train_len_'+str(train_len)
    lstm_prediction_series_file = rootPath + \
        '/prediction/prediction_result/'+resample_train_test_file + \
        '/lstm/'+lstm_prediction_csv
    lstm_prediction_df = pd.read_csv(
        lstm_prediction_series_file, index_col=0, parse_dates=True)
    lstm_prediction_series = pd.Series(
        lstm_prediction_df['0'].values, index=lstm_prediction_df.index)

    return lstm_prediction_series


# 读取测试数据集
resample_train_test_file_dir = get_resample_train_test_file_dir(
    DATA_SET_NAME, TEST_LEN, TRAIN_LEN, RESAMPLE_FREQ_MIN)
test_data_series = get_test_series(resample_train_test_file_dir)

# 读取arima的预测数据集
arima_prediction_series = get_airima_predict_series(
    DATA_SET_NAME, TEST_LEN, TRAIN_LEN, RESAMPLE_FREQ_MIN, pdq_prediction_csv)

lstm_prediction_series = get_lstm_predict_series(
    DATA_SET_NAME, TEST_LEN, TRAIN_LEN, RESAMPLE_FREQ_MIN, lstm_prediction_csv)

arima_mse = mean_squared_error(
    arima_prediction_series.values, test_data_series.values)
lstm_mse = mean_squared_error(
    lstm_prediction_series.values, test_data_series.values)
print('arima error'+str(arima_mse))
print('lstm error'+str(lstm_mse))


static_threshold_pod_num_dic, static_threshold_response_time_dic = static_threshold_method(test_data_series=test_data_series, pod_req_threshold=POD_REQ_THRESHOLD,
                                                                                           pod_cpu_up_threshold=POD_CPU_UP_THRESHOLD, pod_cpu_down_threshold=POD_CPU_DOWN_THRESHOLD,
                                                                                           normal_reqsponse_time=NORMAL_RESPONSE_TIME, more_than_threshold_response_time=MORE_THAN_THRESHOLD_RESPONSE_TIME,
                                                                                           resample_interval_minute=RESAMPLE_FREQ_MIN, pod_intial_minute=POD_INITIAL_MINITE,
                                                                                           pod_first_num=POD_FIRST_NUM)
# test_data_series==prediction_series 预测完全正确
correct_predict_pod_num_dic, correct_predict_response_time_dic = predict(test_data_series=test_data_series, prediction_series=test_data_series, pod_req_threshold=POD_REQ_THRESHOLD,
                                                                         pod_cpu_up_threshold=POD_CPU_UP_THRESHOLD, pod_cpu_down_threshold=POD_CPU_DOWN_THRESHOLD,
                                                                         normal_reqsponse_time=NORMAL_RESPONSE_TIME, more_than_threshold_response_time=MORE_THAN_THRESHOLD_RESPONSE_TIME,
                                                                         resample_interval_minute=RESAMPLE_FREQ_MIN, pod_intial_minute=POD_INITIAL_MINITE,
                                                                         pod_first_num=POD_FIRST_NUM)

arima_predict_pod_num_dic, arima_predict_response_time_dic = predict_with_slow_reduce(test_data_series=test_data_series, prediction_series=arima_prediction_series, pod_req_threshold=POD_REQ_THRESHOLD,
                                                                                      pod_cpu_up_threshold=POD_CPU_UP_THRESHOLD, pod_cpu_down_threshold=POD_CPU_DOWN_THRESHOLD,
                                                                                      normal_reqsponse_time=NORMAL_RESPONSE_TIME, more_than_threshold_response_time=MORE_THAN_THRESHOLD_RESPONSE_TIME,
                                                                                      resample_interval_minute=RESAMPLE_FREQ_MIN, pod_intial_minute=POD_INITIAL_MINITE,
                                                                                      pod_first_num=POD_FIRST_NUM)
lstm_predict_pod_num_dic, lstm_predict_response_time_dic = predict_with_slow_reduce(test_data_series=test_data_series, prediction_series=lstm_prediction_series, pod_req_threshold=POD_REQ_THRESHOLD,
                                                                                    pod_cpu_up_threshold=POD_CPU_UP_THRESHOLD, pod_cpu_down_threshold=POD_CPU_DOWN_THRESHOLD,
                                                                                    normal_reqsponse_time=NORMAL_RESPONSE_TIME, more_than_threshold_response_time=MORE_THAN_THRESHOLD_RESPONSE_TIME,
                                                                                    resample_interval_minute=RESAMPLE_FREQ_MIN, pod_intial_minute=POD_INITIAL_MINITE,
                                                                                    pod_first_num=POD_FIRST_NUM)
print('静态阈值')
caculate_metric(static_threshold_pod_num_dic,
                static_threshold_response_time_dic)
print('预测扩缩_完全正确')
caculate_metric(correct_predict_pod_num_dic, correct_predict_response_time_dic)

print('预测扩缩_arima')
caculate_metric(arima_predict_pod_num_dic, arima_predict_response_time_dic)

print('预测扩缩_lstm')
caculate_metric(lstm_predict_pod_num_dic, lstm_predict_response_time_dic)

sns.set()
fig = plt.figure(figsize=(12, 5))
plt.xlabel('Time(sample every 30s)')

# ax1放pod_num

plt.plot(test_data_series, color='gray',
                label='test_data_series', alpha=0.7, linewidth=1.5)
plt.plot(arima_prediction_series, color='tab:orange',
                label='arima_prediction_series', alpha=0.7, linewidth=1.5)
plt.plot(lstm_prediction_series, color='tab:blue',
                label='lstm_predict', alpha=0.7, linewidth=1.5)

plt.legend()
plt.show()
plt.savefig('prediction', dpi=800,
            bbox_inches='tight')  # transparent=True#














# 绘图resample
# paint_resample_interval = 10
paint_resample_interval_string = str(paint_resample_interval)+'min'

static_threshold_pod_num_series = pd.Series(
    static_threshold_pod_num_dic).resample(paint_resample_interval_string).mean()
correct_predict_pod_num_series = pd.Series(
    correct_predict_pod_num_dic).resample(paint_resample_interval_string).mean()
arima_predict_pod_num_series = pd.Series(arima_predict_pod_num_dic).resample(
    paint_resample_interval_string).mean()
lstm_predict_pod_num_series = pd.Series(lstm_predict_pod_num_dic).resample(
    paint_resample_interval_string).mean()
test_data_series = test_data_series.resample(
    paint_resample_interval_string).mean()


# print('arima: '+str(arima_predict_pod_num_series.values))
# print('lstm: '+str(lstm_predict_pod_num_series.values))
# error=mean_squared_error(arima_predict_pod_num_series.values,lstm_predict_pod_num_series.values)
# print(error)


sns.set()
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(111)
plt.xlabel('Time(resample every' +
           str(int(30*paint_resample_interval/RESAMPLE_FREQ_MIN))+'s)')

# ax1放pod_num
ax1.set_ylabel('Number of Pods')
lns1 = ax1.plot(static_threshold_pod_num_series, color='gray',
                label='static_threshold', alpha=0.7, linewidth=1.5)
lns2 = ax1.plot(lstm_predict_pod_num_series, color='tab:orange',
                label='lstm_predict', alpha=0.7, linewidth=1.5)
lns3 = ax1.plot(arima_predict_pod_num_series, color='tab:blue',
                label='arima_predict', alpha=0.7, linewidth=1.5)

# ax2放请求数量
ax2 = ax1.twinx()
ax2.set_ylabel('Request(requests/s)')
lns4 = ax2.plot(test_data_series, color='green', linestyle='--',
                label='Request', alpha=0.7, linewidth=1.2)
# ax2.tick_params(axis='y')

# 设置图例位置，且背景透明
lns = lns1+lns2+lns3+lns4
labs = [l.get_label() for l in lns]
legend = ax1.legend(lns, labs, bbox_to_anchor=(0.5, -0.2),
                    loc='center', ncol=4, prop={'size': 12}, frameon=False)
frame = legend.get_frame()
frame.set_alpha(1)
frame.set_facecolor('none')  # 设置图例legend背景透明
fig.tight_layout()

# 设置坐标轴区间
ax1.set_ylim(0,)
# ax2.set_ylim(0,)

# 设置去除网格
ax1.grid(False)
ax2.grid(False)

# plt.legend()
plt.show()
plt.savefig('pod_num', dpi=800,
            bbox_inches='tight')  # transparent=True#
