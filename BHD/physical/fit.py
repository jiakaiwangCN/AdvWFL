"""
PL(d) = PL(d_0) + 10 * γ * lgd                                          (n=3)
=> Δγ = ΔPL / (10 * lgd)            (assume d is constant)              (2)
=> γ = ΔPL / (10 * lg(d2 / d1))     (assume γ is constant)              (3)
When we obtain the adversarial sample, we can calculate the amount of change between it and the normal sample.
Based on formula(2), we can calculate the change in γ.
This code fits the γ value corresponding to different layers by equation (3).
After that, we get the layers corresponding to Δγ according to the above fitting results.
"""

"""
对于公式(n=3)，我们计算的是损耗，γ就是损耗因子，显然γ和距离有关，假设γ也和layers有关，那么我们需要对每个(distance, layers)都拟合一个γ，
因为PL(d_0)不可得，所以要通过求Δ的方法计算，所以应该选取同一位置下不同layers的两组值做差，拟合时应该对每组值都拟合出一个γ
"""
import math
import numpy as np
import pandas as pd

VALUE, DISTANCE, LAYERS, DELTA = 0, 1, 2, 3


def get_data(file_name):
    data = pd.read_csv(file_name)
    value = list(data.iloc[:, 0])
    distance = list(data.iloc[:, 1])
    layers = list(data.iloc[:, 2])
    delta = list(data.iloc[:, 3])
    return [[i, j, k, -l] for (i, j, k, l) in zip(value, distance, layers, delta)]


def filter_data(data, key, value):
    return [sublist for sublist in data if sublist[key] == value]


def get_error(data, gamma):
    error = 0
    for item in data:
        error += abs(10 * gamma * math.log10(item[DISTANCE]) - item[DELTA])
    return error / len(data)


def get_best_gamma_and_error(data):
    best_error = 1e5
    best_gamma = 0
    for gamma in candidate:
        error = get_error(data, gamma)
        if best_error > error:
            best_error = error
            best_gamma = gamma
    return [best_gamma, best_error]


if __name__ == "__main__":
    data = get_data('merged.csv')
    candidate = np.arange(-10, 10, 0.0001)
    res = []
    df = pd.DataFrame(columns=['layers', 'distance', 'gamma', 'error'])
    for distance in range(1, 11):
        for layers in range(0, 10):
            res = get_best_gamma_and_error(data=filter_data(filter_data(data, LAYERS, layers), DISTANCE, distance))
            if layers != 0:
                df = df.append({'layers': layers, 'distance': distance, 'gamma': res[0], 'error': res[1]},
                               ignore_index=True)
    df.to_csv('fitted.csv', index=False)

