# coding:utf-8
# @Time: 2022/2/26
# @Author: Heaven

import numpy as np

np.set_printoptions(threshold=np.inf)
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
import LoadData
from sklearn import metrics
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class GAE(Model):
    def __init__(self,v=2,in_dim=32, d=64,rnn_unit=128, horizon=1, **kwargs):
        super(GAE, self).__init__(**kwargs)
        self.code_denses=[]
        self.v=v
        for i in range(v):
            self.code_denses.append(
                keras.layers.LSTM(units=in_dim, return_sequences=True, return_state=True, recurrent_activation='sigmoid', activation='tanh'))
        self.w0 = keras.layers.Dense(d,activation="tanh")
        self.w1 = keras.layers.Dense(d,activation="tanh")
        self.lstm = keras.layers.LSTM(units=rnn_unit, return_sequences=True, return_state=True,
                                      recurrent_activation='sigmoid', activation='tanh')
        self.dense = keras.layers.Dense(horizon)

    def __call__(self, A, matrix, inputs, **kwargs):
        k = 0
        X = []
        for key in inputs.keys():
            sequence_state, _, last_state = self.code_denses[k](inputs[key])
            if k == 0:
                X = sequence_state
            else:
                X = tf.concat([X, sequence_state], axis=1)
            k += 1
        Z = self.w0(tf.matmul(A, X))
        Z = self.w1(tf.matmul(A, Z))

        lstmInput = tf.matmul(matrix, Z)

        sequence_state, _, last_state = self.lstm(lstmInput)

        energy = tf.matmul(sequence_state[:, :-1, :], tf.expand_dims(last_state, -1))
        attention = keras.layers.Activation("softmax")(energy)
        v = tf.reduce_sum(tf.multiply(attention, sequence_state[:, :-1, :]), axis=2)

        attention_state = tf.concat([v, last_state], axis=1)

        out_put = self.dense(attention_state)
        out_put = tf.reshape(out_put, [-1, horizon, 1])

        return out_put

def train_step(tr_X, tr_Y, A, matrix):
    with tf.GradientTape() as g:
        prediction = gae(A, matrix, tr_X, training=True)
        # A = tf.tile(tf.expand_dims(A, axis=0), [A_.shape[0], 1, 1])
        loss_prediction = loss_object(tr_Y, prediction)
        # loss = loss_prediction + tf.keras.losses.categorical_crossentropy(A, A_)
    grads = g.gradient(loss_prediction, gae.trainable_variables)
    optimizer.apply_gradients(zip(grads, gae.trainable_variables))

    train_mae(loss_prediction)
    train_mse(tr_Y, prediction)

# @tf.function
def test_step(te_X, te_Y, A, matrix):
    prediction = gae(A, matrix, te_X, training=False)
    # A = tf.tile(tf.expand_dims(A, axis=0), [A_.shape[0], 1, 1])
    t_loss = loss_object(te_Y, prediction)

    test_mae(t_loss)
    test_mse(te_Y, prediction)

if __name__ == "__main__":

    # path = ["original_data/weather/target.txt", "original_data/weather/lowly.txt", "original_data/weather/highly.txt"]
    # target_path = "original_data/weather/target.txt"
    # m = [1, 1, 6]
    # lag = [24, 24, 144]
    # target_index = 0
    # horizon = 1
    # load = LoadData.LoadWeatherData()
    # train_X, train_Y, test_X, test_Y, dim,A, D, matrix, scaler_test = load.Load_Data(path=path, target_path=target_path, m=m,
    #                                                                             lag=lag, horizon=horizon,
    #                                                                             target_index=target_index)
    #
    # path = ["original_data/stock/quarterly.txt", "original_data/stock/monthly.txt", "original_data/stock/daily.txt"]
    # target_path = "original_data/stock/target.txt"
    # m = [1, 3, 63]
    # lag = [2, 6, 126]
    # target_index = -1
    # horizon = 1
    # load = LoadData.LoadStockData()
    # train_X, train_Y, test_X, test_Y, dim,A,D, matrix, scaler_test = load.Load_Data(path=path, target_path=target_path, m=m,
    #                                                                        lag=lag, horizon=horizon,
    #                                                                        target_index=target_index)

    # path = ["original_data/energy/lowly.txt", "original_data/energy/highly.txt", "original_data/energy/target.txt"]
    # target_path = "original_data/energy/target.txt"
    # m = [1, 3, 3]
    # lag = [3, 6, 6]
    # horizon = 1
    # target_index = -1
    # load = LoadData.LoadEnergyData()
    # train_X, train_Y, test_X, test_Y, dim, A, D, matrix, scaler_test = load.Load_Data(path=path, target_path=target_path, m=m,
    #                                                                        lag=lag, horizon=horizon,
    #                                                                        target_index=target_index)

    m = [64, 32, 1]
    v = 3
    lag = 8
    horizon = 1
    target_index = -1
    load = LoadData.LoadHeartData()
    train_X, train_Y, test_X, test_Y, dim, A, D, matrix, scaler_test = load.Load_Date(m=m, lag=lag)

    degree = np.sum(A, axis=1)
    D_ = np.diag(1 / np.power(degree + 1, 0.5))
    A_ = np.dot(np.dot(D_, A + np.identity(A.shape[0])), D_).astype("float32")
    A = A + np.identity(A.shape[0])

    batch = 512
    train_y = {}
    test_y = {}
    for i in range(len(m)):
        key='input_x'+str(i+1)
        train_X[key] = train_X.pop(i)
        test_X[key] = test_X.pop(i)
    train_y['output'] = train_Y
    test_y['output'] = test_Y

    train_ds = tf.data.Dataset.from_tensor_slices((train_X, train_y)).batch(batch)
    test_ds = tf.data.Dataset.from_tensor_slices((test_X, test_y)).batch(batch)
    progbar = tf.keras.utils.Progbar(len(train_ds), stateful_metrics=['train_loss'])

    loss_object = tf.keras.losses.MeanAbsoluteError()

    train_mae = tf.keras.metrics.Mean(name='train_mse')
    train_mse = tf.keras.metrics.MeanSquaredError(name='train_mae')

    test_mae = tf.keras.metrics.Mean(name='test_mse')
    test_mse = tf.keras.metrics.MeanSquaredError(name='test_mae')

    best_MAE = 100000
    best_MSE = 100000
    best_MAPE = 100000
    mae_list = []
    mse_list = []
    mape_list = []
    best_loss=[]
    for i in range(10):
        loss_list_temp=[]
        gae = GAE(v=len(m), in_dim=50, d=64, rnn_unit=64, horizon=1)

        basetmfcnn = gae
        min_test_loss = 100000
        EPOCHS = 150
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
                train_step(tr_x, tr_Y, A_, matrix)
                progbar.update(i + 1, values=[('train_loss', train_mae.result())])

            for te_x, te_y in test_ds:
                te_batch_y = te_y['output']
                test_step(te_x, te_batch_y, A_, matrix)

            loss_list_temp.append(train_mse.result())
            if test_mae.result() < min_test_loss:
                best_loss = loss_list_temp
                min_test_loss = test_mae.result()
                basetmfcnn = gae

            template = 'Epoch {}, mse: {}, mae: {}, Test mse: {}, Test mae: {}'
            # template = 'Epoch {},  mae: {}, Test mae: {}'
            print(template.format(epoch + 1,
                              train_mse.result(),
                              train_mae.result(),
                              test_mse.result(),
                              test_mae.result()
                                  ))

        test_predict = np.zeros([1, horizon, 1])
        for final_test_x, final_test_y in test_ds:
            prediction = basetmfcnn(A_, matrix, final_test_x, training=False)
            test_predict = np.concatenate((test_predict, prediction), axis=0)

        test_predict = np.reshape(test_predict[1:], [-1, horizon, int(m[-1]/m[target_index])])
        test_true = np.reshape(test_Y, [-1, horizon, int(m[-1]/m[target_index])])

        test_predict = np.mean(test_predict, axis=-1)
        test_true = np.mean(test_true, axis=-1)

        test_predict = np.reshape(test_predict, [-1,horizon])
        test_true = np.reshape(test_true, [-1,horizon])

        # 反归一化
        test_predict = scaler_test.inverse_transform(test_predict)
        test_true = scaler_test.inverse_transform(test_true)

        # plt.plot(range(len(test_true)), test_true)
        # plt.plot(range(len(test_true)), test_predict)
        # plt.show()

        rmse_1 = np.sqrt(metrics.mean_squared_error(y_true=test_true[:, :1], y_pred=test_predict[:, :1]))
        mae_1 = metrics.mean_absolute_error(y_true=test_true[:, :1], y_pred=test_predict[:,:1])
        mape_1 = np.mean(np.abs((test_true[:, :1] - test_predict[:, :1]) / test_true[:, :1]))*100
        mape_list.append(mape_1)
        mae_list.append(mae_1)
        mse_list.append(rmse_1)
        if best_MAE > mae_1:
            best_MAE = mae_1
            best_MSE = rmse_1
            best_MAPE = mape_1
        print(f"horizon=1：rmse: {rmse_1} " f"mae: {mae_1} "  f"mape:{mape_1}")

    loss_file_path = "loss.csv"
    list_name = 'PPG1'
    if not os.path.exists(loss_file_path):
        losspd = pd.DataFrame(best_loss, columns=[list_name])
        losspd.to_csv(loss_file_path, sep=',', index=False)
    else:
        tmpdata = pd.read_csv(loss_file_path)
        tmpdata[list_name] = pd.DataFrame(best_loss)
        tmpdata.to_csv(loss_file_path, sep=',', index=False)

    plt.title("PPG(" + chr(916) + "=1)")
    plt.xlabel("time")
    plt.ylabel("loss")
    plt.plot(best_loss, 'b-', label="True")
    plt.gcf().autofmt_xdate()
    plt.legend(loc='best')
    # plt.savefig('Energy_loss.svg', format='svg')
    plt.show()

    print("mae",mae_list)
    print("rmse",mse_list)
    print("mape",mape_list)
    print("mae_mean:", np.mean(mae_list), "mse_mean:", np.mean(mse_list), "mape_mean:",  np.mean(mape_list))
    print("mae_std:", np.std(mae_list), "mse_std:", np.std(mse_list), "mape_std:",  np.std(mape_list))
    print("best_mae:", best_MAE, "best_mse:", best_MSE, "best_mape:", best_MAPE)
