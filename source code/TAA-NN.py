# coding:utf-8
# @Time: 2020/11/21
# @Author: Heaven

# coding:utf-8
# @Time: 2020/11/3
# @Author: Heaven

import numpy as np

np.set_printoptions(threshold=np.inf)
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from WindowGenerate import WindowGenerator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler,StandardScaler
pd.set_option('display.max_columns', None)
from LoadData import LoadHeartData
from LoadData import LoadCommonData
from sklearn import metrics
from keras import Model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class MFCNN(Model):
    def __init__(self,v_num, rnn_unit, time_pattern_unit=16, context_unit=16, horizon=1):
        super(MFCNN, self).__init__()

        self.v_num = v_num
        self.RNN_Group = []
        self.CNN_Group = []
        self.dim=0
        for i in range(v_num):
            self.RNN_Group.append(
                keras.layers.LSTM(units=rnn_unit[i], return_sequences=True, return_state=True, recurrent_activation='sigmoid', activation='tanh'))
            self.CNN_Group.append(keras.layers.Conv1D(filters=time_pattern_unit, kernel_size=1, padding='valid'))
            self.dim+=rnn_unit[i]

        self.attribute_energy = keras.layers.Dense(time_pattern_unit,activation="relu")
        self.time_energy = keras.layers.Dense(self.dim,activation="relu")

        self.v_dense = keras.layers.Dense(context_unit)
        self.t_dense = keras.layers.Dense(context_unit)
        self.h_dense = keras.layers.Dense(context_unit)

        self.dense = keras.layers.Dense(horizon)

    def call(self, inputs, training=None, mask=None):
        CNN_result = []
        last_state_Set = []
        k=0
        # LSTMs
        for key in inputs.keys():
            sequence_state, _, last_state = self.RNN_Group[k](inputs[key])
            last_state_Set.append(last_state)
            temp_CNN = self.CNN_Group[k](tf.transpose(sequence_state[:, :-1, :], [0, 2, 1]))
            CNN_result.append(temp_CNN)
            k+=1
        #CNNs
        fusion_result = CNN_result[0]
        fusion_last_state = last_state_Set[0]
        for i in range(1, self.v_num):
            fusion_result = tf.concat((fusion_result, CNN_result[i]), axis=1)
            fusion_last_state = tf.concat((fusion_last_state, last_state_Set[i]), axis=1)

        #TAA-NN
        time_energy = tf.matmul(tf.transpose(fusion_result,[0,2,1]), tf.expand_dims(self.time_energy(fusion_last_state), -1))
        attribute_energy = tf.matmul(fusion_result, tf.expand_dims(self.attribute_energy(fusion_last_state), -1))

        time_attention = keras.layers.Activation("softmax")(time_energy)
        attribute_attention = keras.layers.Activation("softmax")(attribute_energy)

        v = tf.reduce_sum(tf.multiply(attribute_attention, fusion_result), axis=1)

        t = tf.reduce_sum(tf.multiply(time_attention, tf.transpose(fusion_result,[0,2,1])), axis=1)

        out_put = self.dense(self.v_dense(v)+self.h_dense(fusion_last_state)+self.t_dense(t))
        out_put=tf.reshape(out_put, [-1, horizon, 1])

        return out_put

def train_step(tr_X,tr_Y):
    with tf.GradientTape() as g:
        prediction = mfcnn(tr_X, training=True)
        loss = loss_object(tr_Y, prediction)
    grads = g.gradient(loss, mfcnn.trainable_variables)
    optimizer.apply_gradients(zip(grads, mfcnn.trainable_variables))

    train_mse(loss)
    train_mae(tr_Y, prediction)

def test_step(te_X, te_Y):
    prediction = mfcnn(te_X, training=False)
    t_loss = loss_object(te_Y, prediction)

    test_mse(t_loss)
    test_mae(te_Y, prediction)

if __name__ == "__main__":
    V_num = 3
    #stock
    # path = ["original_data/stock/quarterly.txt","original_data/stock/monthly.txt","original_data/stock/daily.txt"]
    # target_path = "original_data/stock/daily.txt"
    # con_path = "consequence/stock/H1/s1001.txt"
    # con_loss_path = "consequence/stock/H1/s1001_loss.txt"
    # m = [1, 3, 63]
    # lag = [2, 2, 8]
    # target_index = -1
    # horizon = 1
    # weather
    # path = ["original_data/weather/target.txt", "original_data/weather/lowly.txt", "original_data/weather/highly.txt"]
    # target_path = "original_data/weather/target.txt"
    # m = [1, 1, 6]
    # lag = [24, 24, 24]
    # target_index = 0
    # horizon = 1
    #energy
    path = ["original_data/energy/lowly.txt", "original_data/energy/highly.txt", "original_data/energy/target.txt"]
    target_path = "original_data/energy/target.txt"
    m = [1, 3, 3]
    lag = [3,6,6]
    horizon = 1
    target_index = -1

    #HeartRate
    # m = [64, 32, 1]
    # lag = 8
    # horizon = 1
    # target_index = -1
    # con_path = "consequence/HeartRate/anslist.txt"
    # con_loss_path="consequence/HeartRate/anslist_loss.txt"
    # loadHeart = LoadHeartData()
    # train_X, train_Y, test_X, test_Y, scaler_test = loadHeart.load_Data(m=m, lag=lag)

    loadCommon = LoadCommonData()
    train_X, train_Y, test_X, test_Y, scaler_test = loadCommon.load_Data(path=path,target_path=target_path, m=m, lag=lag,horizon=horizon,target_index=target_index,V_num=V_num)
    batch = 512
    train_y = {}
    test_y = {}
    for i in range(V_num):
        key='input_x'+str(i+1)
        train_X[key] = train_X.pop(i)
        test_X[key] = test_X.pop(i)
    train_y['output'] = train_Y
    test_y['output'] = test_Y

    train_ds = tf.data.Dataset.from_tensor_slices((train_X,train_y)).batch(batch)
    test_ds = tf.data.Dataset.from_tensor_slices((test_X,test_y)).batch(batch)
    progbar = tf.keras.utils.Progbar(len(train_ds), stateful_metrics=['train_loss'])

    loss_object = tf.keras.losses.MeanAbsoluteError()

    train_mse = tf.keras.metrics.Mean(name='train_mse')
    train_mae = tf.keras.metrics.MeanSquaredError(name='train_mae')

    test_mse = tf.keras.metrics.Mean(name='test_mse')
    test_mae = tf.keras.metrics.MeanSquaredError(name='test_mae')
    #stock
    # v_num = V_num, rnn_unit = [16, 16, 8], time_pattern_unit = 32,  context_unit = 32

    #weather
    # v_num = V_num, rnn_unit = [16, 32, 32], time_pattern_unit = 32,  context_unit = 32

    #energy
    # v_num = V_num, rnn_unit = [32, 32, 16], time_pattern_unit = 32, context_unit = 32

    #HeartRate
    # v_num = V_num, rnn_unit = [32, 32, 16], time_pattern_unit = 32, context_unit = 32
    best_MAE = 100000
    best_MSE = 100000
    best_MAPE = 100000
    mae_list = []
    mse_list = []
    mape_list = []
    for i in range(10):
        test_loss_list = []
        mfcnn = MFCNN(v_num=V_num, rnn_unit=[32, 32, 16], time_pattern_unit=32, context_unit=32, horizon=horizon)

        basetmfcnn = mfcnn
        min_test_loss = 100000
        EPOCHS = 200
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        for epoch in range(EPOCHS):
            train_mse.reset_states()
            train_mae.reset_states()
            test_mae.reset_states()
            test_mse.reset_states()

            for i, value in enumerate(train_ds):
                tr_x = value[0]
                tr_y = value[1]
                tr_Y = tr_y['output']
                train_step(tr_x, tr_Y)
                progbar.update(i + 1, values=[('train_loss', train_mae.result())])

            for te_x, te_y in test_ds:
                te_batch_y = te_y['output']
                test_step(te_x, te_batch_y)

            if test_mse.result() < min_test_loss:
                min_test_loss = test_mse.result()
                basetmfcnn = mfcnn
            test_loss_list.append(train_mse.result())
            template = 'Epoch {}, mae: {}, mse: {}, Test mae: {}, Test mse: {}'
            print(template.format(epoch + 1,
                              train_mse.result(),
                              train_mae.result(),
                              test_mse.result(),
                              test_mae.result()))

        test_predict = np.zeros([1, horizon, 1])
        for final_test_x, final_test_y in test_ds:
            prediction = basetmfcnn(final_test_x, training=False)
            test_predict = np.concatenate((test_predict, prediction), axis=0)

        test_predict = np.reshape(test_predict[1:], [-1,horizon,int(m[-1]/m[target_index])])
        test_true = np.reshape(test_Y, [-1,horizon, int(m[-1]/m[target_index])])

        test_predict = np.mean(test_predict, axis=-1)
        test_true = np.mean(test_true, axis=-1)

        test_predict = np.reshape(test_predict, [-1,horizon])
        test_true = np.reshape(test_true, [-1,horizon])

        # inverse
        test_predict = scaler_test.inverse_transform(test_predict)
        test_true = scaler_test.inverse_transform(test_true)


        rmse_1 = np.sqrt(metrics.mean_squared_error(y_true=test_true[:,0], y_pred=test_predict[:,0]))
        mae_1 = metrics.mean_absolute_error(y_true=test_true[:,0], y_pred=test_predict[:,0])
        mape_1 = np.mean(np.abs(test_true[:, 0] - test_predict[:, 0]) / test_true[:, 0]) * 100
        mape_list.append(mape_1)
        mae_list.append(mae_1)
        mse_list.append(rmse_1)
        if best_MAE > mae_1:
            best_MAE = mae_1
            best_MSE = rmse_1
            best_MAPE = mape_1
        print(f"horizon=1：rmse: {rmse_1} " f"mae: {mae_1}"  f"mape:{mape_1}")

        # rmse_3 = np.sqrt(metrics.mean_squared_error(y_true=test_true[:, 2], y_pred=test_predict[:, 2]))
        # mae_3 = metrics.mean_absolute_error(y_true=test_true[:, 2], y_pred=test_predict[:, 2])
        # mape_3 = np.mean(np.abs(test_true[:, 2] - test_predict[:, 2]) / test_true[:, 2]) * 100
        # mape_list.append(mape_3)
        # mae_list.append(mae_3)
        # mse_list.append(rmse_3)
        # if best_MAE > mae_3:
        #     best_MAE = mae_3
        #     best_MSE = rmse_3
        #     best_MAPE = mape_3
        # print(f"horizon=3：rmse: {rmse_3} " f"mae: {mae_3}" f"mape:{mape_3}")

        # rmse_12 = np.sqrt(metrics.mean_squared_error(y_true=test_true[:, -1], y_pred=test_predict[:, -1]))
        # mae_12 = metrics.mean_absolute_error(y_true=test_true[:, -1], y_pred=test_predict[:, -1])
        # mape_12 = np.mean(np.abs(test_true[:,-1]-test_predict[:, -1])/test_true[:,-1])*100
        # mae_list.append(mae_12)
        # mse_list.append(rmse_12)
        # mape_list.append(mape_12)
        # if best_MAE > mae_12:
        #     best_MAE = mae_12
        #     best_MSE = rmse_12
        #     best_MAPE = mape_12
        # print(f"horizon=12：rmse: {rmse_12} " f"mae: {mae_12}" f"mape:{mape_12}")

    print(mae_list)
    print(mse_list)
    print(mape_list)
    print("best_mae:", best_MAE, "best_mse:", best_MSE, "best_mape:", best_MAPE)

