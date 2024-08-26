import csv

import numpy as np
import pandas as pd
import torch

file = pd.read_csv('train.csv')
x = torch.from_numpy(np.array(file.iloc[:, :-2]).astype(np.float32))
y = np.array(file.iloc[:, -2:])

with open('extract_adv_noise_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['RSS'+str(i+1) for i in range(182)]+['x', 'y'])
    for item, cor in zip(x, y):
        # 计算当前数据的直方图
        hist, bin_edges = np.histogram(item, bins=50)

        # 使用直方图生成新的噪声数据
        new_item = np.random.choice(bin_edges[:-1], size=item.shape[0], p=hist / hist.sum())
        data = np.concatenate((new_item, cor))
        writer.writerow(data.tolist())
