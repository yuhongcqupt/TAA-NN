# coding:utf-8
# @Time: 2021/5/19
# @Author: Heaven

import numpy as np

np.set_printoptions(threshold=np.inf)
import pandas as pd

pd.set_option('display.max_columns', None)  #设置显示全部的列
# pd.set_option('display.max_rows', None)  #设置显示所有的行
# import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class LoadHeartData():
    def split_Data(self, BVP, ACC, label, lag, m, target_lag):
        split_X = {}
        for i in range(len(m)):
            split_X[i] = []
        k = target_lag
        for t in range(lag+target_lag*2, int(len(BVP) / 64), 2):
            split_X[0].append(BVP[(t - lag) * m[0]:t * m[0]])
            split_X[1].append(ACC[(t - lag) * m[1]:t * m[1]])
            split_X[2].append(label[k - target_lag:k])
            k += 1
        for i in range(len(m)):
            split_X[i] = np.array(split_X[i]).astype("float32")
        return split_X, np.array(label[lag:]).astype('float32')

    def generateMatrix(self, *arg):
        matrix = np.zeros([arg[0], np.sum(arg)])
        k = 0
        l = 0
        for i in range(arg[0]):
            matrix[i][i] = 1
            if i%2 == 0:
                matrix[i][k+arg[0]] = 1
                k += 1
            if i%64 == 0:
                matrix[i][arg[0]+arg[1]+l] = 1
                l+=1
        return np.array(matrix).astype("float32")

    def getA(self,*args):
        matrix = np.zeros([np.sum(args), np.sum(args)])
        k = 0
        l = 0
        for i in range(args[0]):
            if i%2 == 0:
                matrix[i][k+args[0]] = 1
                matrix[k+args[0]][i] = 1
                k+=1
            if i%64 == 0:
                matrix[i][args[0] + args[1] + l] = 1
                matrix[args[0] + args[1] + l][i] = 1
                l+=1
        return np.array(matrix).astype("float32")


    def getD(self,A):
        size = np.shape(A)[0]
        matrix = np.zeros([size, size])
        for i in range(size):
            matrix[i][i]=np.sum(A[i])
        return np.array(matrix).astype("float32")


    def Load_Date(self, m, lag):
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
                X[k], Y[k]= self.split_Data(BVP[i], ACC[i], label[i], lag, m, target_lag=target_lag)
                if Y[k].shape[0]>X[k][0].shape[0]:
                    Y[k]=Y[k][:-1]
                k+=1

        scaler_test = MinMaxScaler()
        test_BVP = scaler.fit_transform(test_BVP)
        test_ACC = scaler.fit_transform(test_ACC)
        test_label = scaler_test.fit_transform(np.reshape(test_label, [-1, 1]))
        test_X, test_Y = self.split_Data(test_BVP, test_ACC, test_label, lag, m, target_lag=target_lag)
        if test_Y.shape[0] > test_X[0].shape[0]:
            test_Y = test_Y[:-1]

        train_X = {}
        train_X[0] = X[0][0]
        train_X[1] = X[0][1]
        train_X[2] = X[0][2]
        train_Y = Y[0]
        dim=[]
        for i in range(1, len(X)):
            train_X[0] = np.concatenate((train_X[0], X[i][0]),axis=0)
            train_X[1] = np.concatenate((train_X[1], X[i][1]),axis=0)
            train_X[2] = np.concatenate((train_X[2], X[i][2]),axis=0)
            train_Y = np.concatenate((train_Y, Y[i]), axis=0)
        dim.append(train_X[0].shape[2])
        dim.append(train_X[1].shape[2])
        dim.append(train_X[2].shape[2])
        A = self.getA(train_X[0].shape[1],train_X[1].shape[1],train_X[2].shape[1])
        D = self.getD(A)
        matrix=self.generateMatrix(train_X[0].shape[1], train_X[1].shape[1], train_X[2].shape[1])
        return train_X, train_Y, test_X, test_Y, dim, A, D, matrix, scaler_test

class LoadStockData():
    def load_data(self, path, target_path):
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

    def split_data(self, X, Y, lag, m, horizon, target_index):
        split_X={}
        for i in range(len(m)):
            split_X[i] = []
        split_Y = []
        #range(高频数据预测的起点，高频数据长度-向前预测步数，滑动一步)
        for i in range(lag[0] * m[-1], len(X[len(X)-1]) - int(horizon * (m[-1] / m[target_index])), 1):
            # 划分每个频率的数据
            for j in range(len(m)):
                # 计算各频率数据终点
                frequency_end = int((i + 1) * (m[j] / m[-1]))
                split_X[j].append(X[j][frequency_end - lag[j]:frequency_end])
            split_Y.append(
                Y[int((i + 1) * (m[target_index] / m[-1])):int((i + 1) * (m[target_index] / m[-1])) + horizon])
        for i in range(len(m)):
            split_X[i] = np.array(split_X[i]).astype("float32")
        return split_X, np.array(split_Y).astype('float32')

    def generateMatrix(self, args):
        matrix = np.zeros([args[-1], np.sum(args)])
        q=0
        m=0
        for i in range(args[-1]):
            matrix[i][(args[0]+args[1]+i)] = 1
            if i%63 == 0:
                matrix[i][q] = 1
                q+=1
            if i%21 == 0:
                matrix[i][2+m]=1
                m+=1
        return np.array(matrix).astype("float32")

    def getA(self,args):
        matrix = np.zeros([np.sum(args), np.sum(args)])
        q = 0
        m = 0
        for i in range(args[-1]):
            if i%63 == 0:
                matrix[i][q] = 1
                matrix[q][i] = 1
                q+=1
            if i%21 == 0:
                matrix[i][2+m]=1
                matrix[2+m][i] = 1
                m+=1
        return np.array(matrix).astype("float32")

    def getD(self,A):
        size = np.shape(A)[0]
        matrix = np.zeros([size, size])
        for i in range(size):
            matrix[i][i] = np.sum(A[i])
        return np.array(matrix).astype("float32")


    def Load_Data(self, path, target_path, m, lag, horizon, target_index):
        X, Y1, dim = self.load_data(path, target_path)

        scaler = MinMaxScaler()
        scaler_target = MinMaxScaler()
        for i in range(len(m)):
            X[i]=scaler.fit_transform(X[i])
        Y = scaler_target.fit_transform(Y1)

        X, Y = self.split_data(X=X, Y=Y, lag=lag, m=m, horizon=horizon, target_index=target_index)
        train_share = [0.8, 0.2]
        train_X = {}
        test_X = {}
        train_len = int((len(Y1) - int((lag[0] * m[-1] + 1) * (m[target_index] / m[-1]))) * train_share[0]) * int(
            m[-1] / m[target_index])

        for i in range(len(X)):
            train_X[i] = X[i][:train_len]
            test_X[i] = X[i][train_len:]
        train_Y = Y[:train_len]
        test_Y = Y[train_len:]
        matrix=self.generateMatrix(lag)
        A = self.getA(lag)
        D = self.getD(A)
        return train_X, train_Y, test_X, test_Y, dim, A, D, matrix, scaler_target

class LoadWeatherData():
    def load_data(self, path, target_path):
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

    def split_data(self, X, Y, lag, m, horizon, target_index):
        split_X={}
        for i in range(len(m)):
            split_X[i] = []
        split_Y = []
        #range(高频数据预测的起点，高频数据长度-向前预测步数，滑动一步)
        for i in range(lag[0] * m[-1], len(X[len(X)-1]) - int(horizon * (m[-1] / m[target_index])), 1):
            # 划分每个频率的数据
            for j in range(len(m)):
                # 计算各频率数据终点
                frequency_end = int((i + 1) * (m[j] / m[-1]))
                split_X[j].append(X[j][frequency_end - lag[j]:frequency_end])
            split_Y.append(
                Y[int((i + 1) * (m[target_index] / m[-1])):int((i + 1) * (m[target_index] / m[-1])) + horizon])
        for i in range(len(m)):
            split_X[i] = np.array(split_X[i]).astype("float32")
        return split_X, np.array(split_Y).astype('float32')

    def generateMatrix(self, args):
        matrix = np.zeros([args[-1], np.sum(args)])
        q = 0
        for i in range(args[-1]):
            matrix[i][(args[0]+args[1]+i)] = 1
            if i%24 == 0:
                matrix[i][q] = 1
                matrix[i][24+q]=1
                q+=1
        return np.array(matrix).astype("float32")

    def getA(self,args):
        matrix = np.zeros([np.sum(args), np.sum(args)])
        q = 0
        for i in range(args[-1]):
            matrix[i][(args[0] + args[1] + i)] = 1
            if i % 24 == 0:
                matrix[i][q] = 1
                matrix[q][i] = 1
                matrix[24 + q][i] = 1
                matrix[i][24 + q] = 1
                q += 1
        return np.array(matrix).astype("float32")

    def getD(self,A):
        size = np.shape(A)[0]
        matrix = np.zeros([size, size])
        for i in range(size):
            matrix[i][i] = np.sum(A[i])
        return np.array(matrix).astype("float32")

    def Load_Data(self, path, target_path, m, lag, horizon, target_index):
        X, Y1, dim = self.load_data(path, target_path)

        scaler = MinMaxScaler()
        scaler_target = MinMaxScaler()
        for i in range(len(m)):
            X[i]=scaler.fit_transform(X[i])
        Y = scaler_target.fit_transform(Y1)

        X, Y = self.split_data(X=X, Y=Y, lag=lag, m=m, horizon=horizon, target_index=target_index)
        train_share = [0.8, 0.2]
        train_X = {}
        test_X = {}
        train_len = int((len(Y1) - int((lag[0] * m[-1] + 1) * (m[target_index] / m[-1]))) * train_share[0]) * int(
            m[-1] / m[target_index])
        for i in range(len(X)):
            train_X[i] = X[i][:train_len]
            test_X[i] = X[i][train_len:]
        train_Y = Y[:train_len]
        test_Y = Y[train_len:]
        matrix = self.generateMatrix(lag)
        A = self.getA(lag)
        D = self.getD(A)
        return train_X, train_Y, test_X, test_Y, dim,A,D, matrix, scaler_target

class LoadEnergyData():
    def load_data(self, path, target_path):
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

    def split_data(self, X, Y, lag, m, horizon, target_index):
        split_X={}
        matrix=[]
        for i in range(len(m)):
            split_X[i] = []
        split_Y = []
        #range(高频数据预测的起点，高频数据长度-向前预测步数，滑动一步)
        for i in range(lag[0] * m[-1], len(X[len(X)-1]) - int(horizon * (m[-1] / m[target_index])), 1):
            # 划分每个频率的数据
            for j in range(len(m)):
                # 计算各频率数据终点
                frequency_end = int((i + 1) * (m[j] / m[-1]))
                split_X[j].append(X[j][frequency_end - lag[j]:frequency_end])
            split_Y.append(Y[int((i + 1) * (m[target_index] / m[-1])):int((i + 1) * (m[target_index] / m[-1])) + horizon])
            matrix.append(np.zeros([lag[-1], np.sum(lag)]))

        for i in range(len(m)):
            split_X[i] = np.array(split_X[i]).astype("float32")
        return split_X, np.array(split_Y).astype('float32')

    def generateMatrix(self, lag):
        matrix = np.zeros([lag[-1], np.sum(lag)])
        q = 0
        for i in range(lag[-1]):
            matrix[i][(lag[0]+lag[1]+i)] = 1
            matrix[i][(lag[0]+i)] = 1
            if i%2 == 0:
                matrix[i][q] = 1
                q+=1
        return np.array(matrix).astype("float32")

    def getA(self,args):
        matrix = np.zeros([np.sum(args), np.sum(args)])
        q = 0
        for i in range(args[-1]):
            if i % 2 == 0:
                matrix[i][q] = 1
                matrix[q][i] = 1
                q += 1
        return np.array(matrix).astype("float32")

    def getD(self,A):
        size = np.shape(A)[0]
        matrix = np.zeros([size, size])
        for i in range(size):
            matrix[i][i] = np.sum(A[i])
        return np.array(matrix).astype("float32")

    def Load_Data(self, path, target_path, m, lag, horizon, target_index):
        X, Y1, dim = self.load_data(path, target_path)

        scaler = MinMaxScaler()
        scaler_target = MinMaxScaler()
        for i in range(len(m)):
            X[i]=scaler.fit_transform(X[i])
        Y = scaler_target.fit_transform(Y1)

        X, Y = self.split_data(X=X, Y=Y, lag=lag, m=m, horizon=horizon, target_index=target_index)
        train_share = [0.8, 0.2]
        train_X = {}
        test_X = {}
        train_len = int((len(Y1) - int((lag[0] * m[-1] + 1) * (m[target_index] / m[-1]))) * train_share[0]) * int(
            m[-1] / m[target_index])
        for i in range(len(X)):
            train_X[i] = X[i][:train_len]
            test_X[i] = X[i][train_len:]
        train_Y = Y[:train_len]
        test_Y = Y[train_len:]
        matrix = self.generateMatrix(lag)
        A = self.getA(lag)
        D = self.getD(A)
        return train_X, train_Y, test_X, test_Y, dim, A, D, matrix, scaler_target

# if __name__ == "__main__":
#     load = LoadHeartData()
#     train_X, train_Y, test_X, test_Y,matrix , scaler_test = load.Load_Date()


    # path = ["original_data/stock/quarterly.txt", "original_data/stock/monthly.txt", "original_data/stock/daily.txt"]
    # target_path = "original_data/stock/target.txt"
    # m = [1, 3, 63]
    # lag = [2, 6, 126]
    # target_index = -1
    # horizon = 1
    # load = LoadStockData()
    # train_X, train_Y, test_X, test_Y,dim, matrix, scaler_test = load.Load_Data(path=path, target_path=target_path, m=m, lag=lag,horizon=horizon,target_index=target_index)

    # path = ["original_data/weather/target.txt", "original_data/weather/lowly.txt", "original_data/weather/highly.txt"]
    # target_path = "original_data/weather/target.txt"
    # m = [1, 1, 6]
    # lag = [24, 24, 144]
    # target_index = 0
    # horizon = 1
    # load = LoadWeatherData()
    # train_X, train_Y, test_X, test_Y,dim, matrix, scaler_test = load.Load_Data(path=path, target_path=target_path, m=m,
    #                                                                        lag=lag, horizon=horizon,
    #                                                                        target_index=target_index)