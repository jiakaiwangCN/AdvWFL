import time
from math import sqrt

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset


def one_hot_encode(file_path):
    file = pd.read_csv(file_path)
    one_hot_dic = {}
    y = np.array(file.iloc[:, -2:])
    one_hot_dic = {}
    for i in y:
        one_hot_dic[(i[0], i[1])] = len(one_hot_dic)
    inverse_dic = {}
    for i, key in enumerate(one_hot_dic):
        inverse_dic[i + 1] = key
    return one_hot_dic, inverse_dic


# 输入
def get_error(predict, abs, dic):
    predict = torch.nn.functional.softmax(predict, dim=1)
    predict = predict.detach().numpy()
    abs = abs.detach().numpy()
    res = []
    for i in range(len(abs)):
        temp = dic[np.argmax(abs[i]) + 1]
        x_abs, y_abs = temp[0], temp[1]
        x_predict = 0.0
        y_predict = 0.0
        for j in range(len(predict[i])):
            temp = dic[j + 1]
            x_predict += predict[i][j] * temp[0]
            y_predict += predict[i][j] * temp[1]
        res.append(sqrt((x_abs - x_predict) ** 2 + (y_abs - y_predict) ** 2))
    return res


def get_coordinate(predict, dic):
    predict = torch.nn.functional.softmax(predict, dim=0)
    x_predict = 0.0
    y_predict = 0.0
    for j in range(len(predict)):
        temp = dic[j + 1]
        x_predict += predict[j] * temp[0]
        y_predict += predict[j] * temp[1]
    return float(x_predict), float(y_predict)


class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, input, target, dic):
        return get_error(input, target, dic)


class MyDataset(Dataset):
    def __init__(self, file_path, dic):
        file = pd.read_csv(file_path)
        self.x = torch.from_numpy(np.array(file.iloc[:, :-2]).astype(np.float32))
        self.y = np.array(file.iloc[:, -2:])
        # for i in y:
        #     temp = [0] * len(dic)
        #     temp[dic[(i[0], i[1])] - 1] = 1
        #     self.y.append(temp)
        # self.y = torch.from_numpy(np.array(self.y).astype(np.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item]


class AdvDataset(Dataset):
    def __init__(self, train_batch, attack):
        self.x = []
        self.y = []
        t1 = time.perf_counter()
        for step, data in enumerate(train_batch):
            print("\rGenerating data {} / {} batch".format(step + 1, len(train_batch)), end="")
            x, y = attack.generate_data(data)
            self.x.append(x)
            self.y.append(y)
        self.x = torch.cat(self.x, dim=0)
        self.y = torch.cat(self.y, dim=0)
        print("\nUse {} s\n".format(time.perf_counter() - t1))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item]


def get_asr(normal, attack, threshold):
    asr = 0
    for i, j in zip(normal, attack):
        if j >= i + threshold:
            asr += 1
    return asr / len(normal)