# univariate multi-step vector-output stacked lstm example


from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import sys
import os
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
from keras.models import load_model
import pandas as pd
from datetime import timedelta
from sklearn.metrics import mean_squared_error
import seaborn as sns

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
sys.path.append(rootPath)

from data.handle_data_set2 import read_dataset_by_name, split, get_train_series, get_resample_train_test_file_dir, get_test_series
from data.handle_data_set2 import DATA_SET_NAME, RESAMPLE_FREQ_MIN, TEST_LEN, RESAMPLE_FREQ_MIN_STRING, TRAIN_LEN
from my_common.utils import get_logger
# 构建(多步+单变量输入)_(多步+单变量输出)


def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # 找到输入的结束位置
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # 找到输出的结束位置，输出=输入结束位置+输出长度
        if out_end_ix > len(sequence):
            break
        # 根据计算的位置获取输入输出数据
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def train(n_steps_in, n_steps_out, n_features, X_train, y_train, X_test, y_test,resample_train_test_file_dir):

    # define model
    model = Sequential()
    # 定义输入的格式input_shape为(3,1),因此在fit()时，传入X(5,3,1),y(5,2)，模型就会明白这是5组输入输出对
    model.add(LSTM(30, activation='relu', return_sequences=True,
                   input_shape=(n_steps_in, n_features)))
    model.add(LSTM(30, activation='relu'))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')

    model_save_file='model_file/lstm/'+resample_train_test_file_dir
    
    if not os.path.exists(os.path.join(model_save_file)):
        os.makedirs(os.path.join(model_save_file))

    checkpoint = ModelCheckpoint(filepath=model_save_file+'/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto',
                                 period=1
                                 )
    # fit model
    history = model.fit(X_train, y_train, epochs=3000, verbose=1, callbacks=[
                        checkpoint], validation_data=(X_test, y_test))
    # 这个validation_data好像不对劲
    # history=model.fit(X_train, y_train, epochs=3000,verbose=1,callbacks=[checkpoint])

    fig = plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.show()
    plt.legend()
    fig.savefig('train.png')
# 构造一条符合要求的输入数据进行测试,将待预测序列x_input(3,)转换成x_input(1,3,1),1表示每批传入1组数据，3表示时间步，1表示特征


def test(n_steps_in, n_steps_out, n_features, X_train, y_train, X_test, y_test, model_file, test_data_series, train_data_series,resample_train_test_file_dir):
    model_path ='model_file/lstm/'+resample_train_test_file_dir+'/'+model_file+'.h5'
    model = load_model(model_path)
    # x_input=array([18,18,18,19])
    # x_input = x_input.reshape((1, n_steps_in, n_features))
    predictions_series = pd.Series([], dtype='float64')
    data_series = pd.concat([train_data_series, test_data_series])
    for i in range(n_steps_in+1):
        time = test_data_series.index[i]
        temp_array = data_series[time-timedelta(minutes=RESAMPLE_FREQ_MIN*(
            n_steps_out+n_steps_in-1)):time-timedelta(minutes=RESAMPLE_FREQ_MIN*(n_steps_out))].values
        temp_array = temp_array.reshape((1, n_steps_in, n_features))
        yhat = model.predict(temp_array, verbose=0)
        predictions_series[test_data_series.index[i]] = yhat[0][1]
    # test_data_series[n_steps_in+1:]
    yhat = model.predict(X_test, verbose=0)
    for i in range(len(yhat)):
        predictions_series[test_data_series.index[n_steps_in+1+i]] = yhat[i][1]

    error = mean_squared_error(
        predictions_series.values, test_data_series.values)

    csv_save_path = os.path.join(
        'prediction_result/' + resample_train_test_file_dir+'/lstm')
    fig_save_path = os.path.join(
        'prediction_result/' + resample_train_test_file_dir+'/lstm/fig')

    if not os.path.exists(csv_save_path):
        os.makedirs(csv_save_path)

    if not os.path.exists(fig_save_path):
        os.makedirs(fig_save_path)

    predictions_series.to_csv(csv_save_path+'/'+model_file+'_prediction.csv')
    sns.set()
    fig = plt.figure(figsize=(15, 5))
    plt.plot(test_data_series, label='test')
    plt.plot(predictions_series, label='prediction')

    plt.legend()
    plt.savefig(fig_save_path+'/'+model_file+'_prediction.png', dpi=400,
                bbox_inches='tight')  # transparent=True#

    print(error)

def paint_network( model_file,resample_train_test_file_dir):
    model_path ='model_file/lstm/'+resample_train_test_file_dir+'/'+model_file+'.h5'
    model = load_model(model_path)
    #只需要有网络结构就可以，不需要进行训练。
    #绘制网络结构
    import pydot_ng as pydot
    from keras.utils.vis_utils import plot_model#自带的绘图的包
    plot_model(model,to_file="model_plot_model.png",show_shapes=True,show_layer_names='False',rankdir='TB')
    plt.figure(figsize=(10,10))
    img=plt.imread("model_plot_model.png")
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # import visualkeras
    # from PIL import ImageFont
    # import matplotlib.pyplot as plt
    # # font = ImageFont.truetype("arial.ttf", 80)
    # plt.imshow(visualkeras.layered_view(model, legend=True))
    # visualkeras.layered_view(model, legend=True).save('model.png')
    # plt.axis('off')
    # plt.show()
    # # visualkeras.layered_view(model)
    # # visualkeras.graph_view(model)

    from keras_visualizer import visualizer  
    plt.imshow(visualizer(model, format='png', view=True))
    visualizer(model, format='png', view=True).save('model_visualizer.png')
    
    # from ann_visualizer.visualize import ann_viz;
    # ann_viz(model, title="").save('model.png')
def main():
    # define input sequence
    # 读取训练和测试数据集
    resample_train_test_file_dir = get_resample_train_test_file_dir(
        DATA_SET_NAME, TEST_LEN, TRAIN_LEN, RESAMPLE_FREQ_MIN)
    train_data_series = get_train_series(resample_train_test_file_dir)
    test_data_series = get_test_series(resample_train_test_file_dir)
    train_seq = train_data_series.values
    test_seq = test_data_series.values

    # 输入数据的步长为3，输出数据的步长为2
    n_steps_in, n_steps_out = 4, 2
    # 数据集dataset(9,3)变成输入输出对：X(5,3),y(5,2)
    X_train, y_train = split_sequence(train_seq, n_steps_in, n_steps_out)
    X_test, y_test = split_sequence(test_seq, n_steps_in, n_steps_out)

    # 将X(5,3)转成X(5,3,1),表示共5组数据，每组3个步长，每个步长1个特征值
    n_features = 1
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))

    # train(n_steps_in,n_steps_out,n_features,X_train,y_train,X_test,y_test,resample_train_test_file_dir)
    model_file = 'ep1102-loss2.877-val_loss3.416'
    # test(n_steps_in, n_steps_out, n_features, X_train, y_train, X_test,
    #      y_test, model_file, test_data_series, train_data_series,resample_train_test_file_dir)

    paint_network(model_file,resample_train_test_file_dir)
if __name__ == "__main__":
    main()
