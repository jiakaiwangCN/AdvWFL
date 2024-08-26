import csv
import os

import numpy as np
import pandas as pd
import torch


def add_gauss_noise(item):
    shape = item.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, shape[0])
    gauss = gauss.reshape(shape)
    return item + gauss


def generate_gauss_noise(file_name):
    file = pd.read_csv(file_name)
    x = torch.from_numpy(np.array(file.iloc[:, :-3]).astype(np.float32))
    y = np.array(file.iloc[:, -3:])

    with open(os.path.join(os.path.dirname(file_name), 'gauss_data.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['RSS' + str(i + 1) for i in range(992)] + ['x', 'y', 'z'])
        for item, cor in zip(x, y):
            gauss_item = add_gauss_noise(item)
            data = np.concatenate((gauss_item.numpy(), cor))
            writer.writerow(data.tolist())

    return os.path.join(os.path.dirname(file_name), 'gauss_data.csv')
