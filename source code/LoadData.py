# coding:utf-8
# @Time: 2021/1/6
# @Author: Heaven

import numpy as np

np.set_printoptions(threshold=np.inf)
import pandas as pd

pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.preprocessing import MinMaxScaler,StandardScaler

class LoadHeartData():
    def split_Data(self, BVP_, ACC_, label_, lag, m, target_lag):
        split_X = {}
        for i in range(len(m)):
            split_X[i] = []
        k=target_lag
        for t in range(lag+target_lag*2, int(len(BVP_) / 64), 2):
            split_X[0].append(BVP_[(t-lag) * m[0]:t * m[0]])
            split_X[1].append(ACC_[(t-lag) * m[1]:t * m[1]])
            split_X[2].append(label_[k-target_lag:k])
            k+=1
        for i in range(len(m)):
            split_X[i] = np.array(split_X[i]).astype("float32")
        return split_X, np.array(label_[lag:]).astype('float32')

    def split_downsampling_Data(self, X_, label_, lag, m):
        split_X = []
        for t in range(0, int(len(X_) / 32) - lag, 2):
            split_X.append(X_[t * m[0]:(t + lag) * m[0]])
        return np.array(split_X).astype('float32'), np.array(label_).astype('float32')

    def split_AR_Data(self, X, lag):
        split_x=[]
        split_y=[]
        for i in range(len(X)-lag):
            split_x.append(X[i: i+lag])
            split_y.append(X[i+lag])
        return np.array(split_x).astype('float32'),np.array(split_y).astype('float32')

    def load_Data(self, m, lag):
        with open(r'original_data/HeartRate/BVP.pkl', 'rb') as f:
            BVP = pickle.load(f, encoding='iso-8859-1')
        with open(r'original_data/HeartRate/ACC.pkl', 'rb') as f:
            ACC = pickle.load(f, encoding='iso-8859-1')
        with open(r'original_data/HeartRate/label.pkl', 'rb') as f:
            label = pickle.load(f, encoding='iso-8859-1')
        test_index = 3
        test_BVP = BVP.pop(test_index)
        test_ACC = ACC.pop(test_index)
        test_label = label.pop(test_index)
        X = {}
        Y = {}
        scaler = MinMaxScaler()
        k=0
        target_lag = 8
        for i in range(10):
            if i!=test_index:
                BVP[i] = scaler.fit_transform(BVP[i])
                ACC[i] = scaler.fit_transform(ACC[i])
                label[i] = scaler.fit_transform(np.reshape(label[i], [-1, 1]))
                X[k], Y[k] = self.split_Data(BVP[i], ACC[i], label[i], lag, m,target_lag=target_lag)
                if Y[k].shape[0]>X[k][0].shape[0]:
                    Y[k]=Y[k][:-1]
                k+=1

        scaler_test = MinMaxScaler()
        test_BVP = scaler.fit_transform(test_BVP)
        test_ACC = scaler.fit_transform(test_ACC)
        test_label = scaler_test.fit_transform(np.reshape(test_label, [-1, 1]))
        test_X, test_Y = self.split_Data(test_BVP, test_ACC, test_label, lag, m,target_lag=target_lag)
        if test_Y.shape[0] > test_X[0].shape[0]:
            test_Y = test_Y[:-1]

        train_X = {}
        train_X[0] = X[0][0]
        train_X[1] = X[0][1]
        train_X[2] = X[0][2]
        train_Y = Y[0]
        for i in range(1, len(X)):
            train_X[0] = np.concatenate((train_X[0],X[i][0]),axis=0)
            train_X[1] = np.concatenate((train_X[1],X[i][1]),axis=0)
            train_X[2] = np.concatenate((train_X[2],X[i][2]),axis=0)
            train_Y = np.concatenate((train_Y, Y[i]),axis=0)
        return train_X, train_Y, test_X, test_Y, scaler_test

    def MIDAS_load_Data(self,m,lag):
        with open(r'original_data/HeartRate/BVP.pkl', 'rb') as f:
            BVP = pickle.load(f, encoding='iso-8859-1')
        with open(r'original_data/HeartRate/ACC.pkl', 'rb') as f:
            ACC = pickle.load(f, encoding='iso-8859-1')
        with open(r'original_data/HeartRate/label.pkl', 'rb') as f:
            label = pickle.load(f, encoding='iso-8859-1')
        # test_index = random.randrange(10)
        test_index = 3
        test_BVP = BVP.pop(test_index)
        test_ACC = ACC.pop(test_index)
        test_label = label.pop(test_index)
        X = {}
        Y = {}
        scaler = MinMaxScaler()
        k = 0
        target_lag = 8
        for i in range(10):
            if i != test_index:
                BVP[i] = scaler.fit_transform(BVP[i])
                ACC[i] = scaler.fit_transform(ACC[i])
                label[i] = scaler.fit_transform(np.reshape(label[i], [-1, 1]))
                X[k], Y[k] = self.split_Data(BVP[i], ACC[i], label[i], lag, m, target_lag=target_lag)
                if Y[k].shape[0] > X[k][0].shape[0]:
                    Y[k] = Y[k][:-1]
                k += 1
        scaler_test = MinMaxScaler()
        test_BVP = scaler.fit_transform(test_BVP)
        test_ACC = scaler.fit_transform(test_ACC)
        test_label = scaler_test.fit_transform(np.reshape(test_label, [-1, 1]))
        t_X, t_Y = self.split_Data(test_BVP, test_ACC, test_label, lag, m, target_lag=target_lag)
        if t_Y.shape[0] > t_X[0].shape[0]:
            t_Y = t_Y[:-1]

        test_X = {}
        test_X[0] = t_X[0]
        test_X[1] = np.expand_dims(t_X[1][:,:,0], -1)
        test_X[2] = np.expand_dims(t_X[1][:,:,1], -1)
        test_X[3] = np.expand_dims(t_X[1][:,:,2], -1)
        test_X[4] = t_X[2]
        test_Y = t_Y

        train_X = {}
        train_X[0] = X[0][0]
        train_X[1] = np.expand_dims(X[0][1][:,:,0], -1)
        train_X[2] = np.expand_dims(X[0][1][:,:,1], -1)
        train_X[3] = np.expand_dims(X[0][1][:,:,2], -1)
        train_X[4] = X[0][2]
        train_Y = Y[0]
        for i in range(1, len(X)):
            train_X[0] = np.concatenate((train_X[0],X[i][0]),axis=0)
            train_X[1] = np.concatenate((train_X[1],np.expand_dims(X[i][1][:,:,0],-1)),axis=0)
            train_X[2] = np.concatenate((train_X[2],np.expand_dims(X[i][1][:,:,1],-1)),axis=0)
            train_X[3] = np.concatenate((train_X[3],np.expand_dims(X[i][1][:,:,2],-1)),axis=0)
            train_X[4] = np.concatenate((train_X[4],X[i][2]),axis=0)
            train_Y = np.concatenate((train_Y, Y[i]),axis=0)

        return train_X, train_Y, test_X, test_Y, scaler_test

    def downsampling_load_Data(self):
        with open(r'original_data/HeartRate/BVP.pkl', 'rb') as f:
            BVP = pickle.load(f, encoding='iso-8859-1')
        with open(r'original_data/HeartRate/ACC.pkl', 'rb') as f:
            ACC = pickle.load(f, encoding='iso-8859-1')
        with open(r'original_data/HeartRate/label.pkl', 'rb') as f:
            label = pickle.load(f, encoding='iso-8859-1')
        down_BVP = {}
        lag = 8
        m = [32, 32]
        for i in range(len(BVP)):
            temp=np.zeros([int(BVP[i].shape[0]/2),BVP[i].shape[1]])
            k=0
            for j in range(1, len(BVP[i]), 2):
                temp[k] = (BVP[i][j-1]+BVP[i][j])/2
                k+=1
            down_BVP[i] = temp

        concat_X={}
        for i in range(len(BVP)):
            concat_X[i] = np.concatenate((down_BVP[i],ACC[i]), axis=1)

        X={}
        Y={}
        test_index = 3
        test_X = concat_X.pop(test_index)
        test_label = label.pop(test_index)
        scaler = MinMaxScaler()
        k=0
        for i in range(len(concat_X)):
            if i != test_index:
                concat_X[i]=scaler.fit_transform(concat_X[i])
                label[i]=scaler.fit_transform(np.reshape(label[i],[-1,1]))
                X[k], Y[k] = self.split_downsampling_Data(concat_X[i], label[i], lag, m)
                if Y[k].shape[0] > X[k].shape[0]:
                    Y[k] = Y[k][:-1]
                k += 1
        scaler_test = MinMaxScaler()
        test_X = scaler.fit_transform(test_X)
        test_label = scaler_test.fit_transform(np.reshape(test_label,[-1,1]))
        test_X, test_Y = self.split_downsampling_Data(test_X, test_label, lag, m)
        if test_Y.shape[0] > test_X.shape[0]:
            test_Y = test_Y[:-1]

        train_X = X[0]
        train_Y = Y[0]
        for i in range(1, len(X)):
            train_X = np.concatenate((train_X, X[i]), axis=0)
            train_Y = np.concatenate((train_Y, Y[i]), axis=0)
        return train_X, train_Y, test_X, test_Y, scaler_test

    def AR_load_Data(self, lag):
        with open(r'original_data/HeartRate/label.pkl', 'rb') as f:
            label = pickle.load(f, encoding='iso-8859-1')

        scaler_test = MinMaxScaler()
        test_index = 3
        test_Y = np.reshape(np.array(label.pop(test_index)).astype("float32"),[-1,1])
        test_Y = scaler_test.fit_transform(test_Y)
        test_X,test_Y=self.split_AR_Data(test_Y,lag)

        scaler = MinMaxScaler()
        k=0
        X={}
        Y={}
        for i in range(len(label)+1):
            if(i!=test_index):
                temp=scaler.fit_transform(np.reshape(np.array(label[i]),[-1,1]))
                X[k], Y[k]=self.split_AR_Data(temp,lag)
                k+=1
        train_X = X[0]
        train_Y = Y[0]
        for i in range(1, len(X)):
            train_X=np.concatenate((train_X, X[i]), axis=0)
            train_Y=np.concatenate((train_Y, Y[i]), axis=0)
        return train_X, train_Y, test_X, test_Y, scaler_test

class LoadCommonData():
    #本文
    def load_df(self,path, target_path):
        V_num = len(path)
        df = {}
        data = {}
        dim = []
        for i in range(V_num):
            df[i] = pd.read_csv(path[i], header=None, sep="\t")
            data[i] = np.array(df[i]).astype('float32')
            dim.append(data[i].shape[1])
        target = pd.read_csv(target_path, header=None, sep="\t")
        target = np.array(target).astype('float32')
        return data, target, dim

    def split_Data(self,X, Y, lag, m, horizon, target_index):
        split_X = {}
        for i in range(len(m)):
            split_X[i] = []
        data_Y = []
        for i in range(lag[0] * m[-1], len(X[len(X) - 1]) - int(horizon * (m[-1] / m[target_index])), 1):
            for j in range(len(m)):
                frequency_start = int((i + 1) * (m[j] / m[-1]))
                split_X[j].append(X[j][frequency_start - lag[j]:frequency_start])
            data_Y.append(
                Y[int((i + 1) * (m[target_index] / m[-1])):int((i + 1) * (m[target_index] / m[-1])) + horizon])
        for i in range(len(m)):
            split_X[i] = np.array(split_X[i]).astype("float32")
        return split_X, np.array(data_Y).astype('float32')

    def load_Data(self,path,target_path, m, lag,horizon,target_index,V_num):
        X, Y, dim = self.load_df(path, target_path)
        diff_X = {}

        scaler = MinMaxScaler()
        scaler_test = MinMaxScaler()
        for i in range(V_num):
            diff_X[i] = scaler.fit_transform(X[i])
        diff_Y = scaler_test.fit_transform(Y)

        X, diff_Y = self.split_Data(X=diff_X, Y=diff_Y, lag=lag, m=m, horizon=horizon, target_index=target_index)

        train_share = [0.80, 0.2]
        train_X = {}
        test_X = {}
        train_len = int((len(Y) - int((lag[0] * m[-1] + 1) * (m[target_index] / m[-1]))) * train_share[0]) * int(
            m[-1] / m[target_index])
        for i in range(len(X)):
            train_X[i] = X[i][:train_len]
            test_X[i] = X[i][train_len:]
        train_Y = diff_Y[:train_len]
        test_Y = diff_Y[train_len:]
        return train_X, train_Y, test_X, test_Y, scaler_test
    #MIDAS
    def MIDAS_load_df(self,path, target_path):
        V_num = len(path)
        data = {}
        for i in range(V_num):
            data[i] = np.array(pd.read_csv(path[i], header=None, sep="\t")).astype('float32')
        target = np.array(pd.read_csv(target_path, header=None, sep="\t")).astype('float32')

        X = {}
        z = 0
        for i in range(V_num):
            for j in range(data[i].shape[1]):
                X[z] = data[i][:, j]
                z += 1

        return X, target

    def MIDAS_split_Data(self,X, Y, lag, m, horizon, target_index):
        split_X = {}
        for i in range(len(m)):
            split_X[i] = []
        data_Y = []
        for i in range(lag[0] * m[-1], len(X[len(X) - 1]) - int(horizon * (m[-1] / m[target_index])), 1):
            for j in range(len(m)):
                frequency_start = int((i + 1) * (m[j] / m[-1]))
                split_X[j].append(X[j][frequency_start - lag[j]:frequency_start])
            data_Y.append(
                Y[int((i + 1) * (m[target_index] / m[-1])):int((i + 1) * (m[target_index] / m[-1])) + horizon])
        for i in range(len(m)):
            split_X[i] = np.array(split_X[i]).astype("float32")
        return split_X, np.array(data_Y).astype('float32')

    def MIDAS_load_Data(self,path,target_path, m, lag,horizon,target_index):
        X, target = self.MIDAS_load_df(path, target_path)

        scaler = MinMaxScaler()
        for i in range(len(X)):
            X[i] = np.reshape(X[i], [-1, 1])
            X[i] = scaler.fit_transform(X[i])
        scaler_test = MinMaxScaler()
        target = scaler_test.fit_transform(target)

        X, Y = self.MIDAS_split_Data(X=X, Y=target, lag=lag, m=m, horizon=horizon, target_index=target_index)
        train_share=[0.8,0.2]
        train_len = int((len(target) - int((lag[0] * m[-1] + 1) * (m[target_index] / m[-1]))) * train_share[0]) * int(
            m[-1] / m[target_index])
        train_X = {}
        test_X = {}
        for i in range(len(X)):
            train_X[i] = X[i][:train_len]
            test_X[i] = X[i][train_len:]
        train_Y = Y[:train_len]
        test_Y = Y[train_len:]
        return train_X, train_Y, test_X, test_Y, scaler_test
    #AR
    def AR_load_df(self,target_path):
        target = pd.read_csv(target_path, header=None, sep="\t")
        target = np.array(target).astype('float32')
        return target

    def AR_split_Data(self,Y, lag):
        X = []
        data_Y = []
        for i in range(len(Y) - lag):
            X.append(Y[i:i + lag])
            data_Y.append(Y[i + lag])
        return np.array(X).astype('float32'), np.array(data_Y).astype('float32')

    def AR_load_Data(self,target_path, lag):
        Y = self.AR_load_df(target_path)
        scalerY = MinMaxScaler()
        diff_Y = scalerY.fit_transform(Y)

        diff_X, diff_Y = self.AR_split_Data(diff_Y, lag)

        train_share = [0.80, 0.2]
        train_len = int(len(diff_Y) * train_share[0])
        train_X = diff_X[:train_len]
        test_X = diff_X[train_len:]

        train_Y = diff_Y[:train_len]
        test_Y = diff_Y[train_len:]
        return train_X,train_Y,test_X,test_Y,scalerY


