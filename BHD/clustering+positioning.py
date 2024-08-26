# Copyright (c) 2017 Tampere University of Technology (TUT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# python script to cluster WLAN RSS fingerprint data with affinity projection
# method and compute positioning error on test data
import numpy as np
from numpy import genfromtxt
from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering
from sklearn.decomposition import PCA
import time
import pandas as pd

from my_data import get_asr

# enter path to directory where data is stored
path_to_database = './classic_data'
# choose algorithm 'km' for k-means and 'ap' for affinity propagation
method = 'ap'  # 'km'


def load_data(path_to_data):
    # training data
    df = pd.read_csv(path_to_data + '/train.csv')

    # 选择前182列，保存到test_rssi.csv
    df_rssi = df.iloc[:, :182]
    df_rssi.to_csv(path_to_data + '/train_rssi.csv', index=False)

    # 选择最后两列，保存到test_cor.csv
    df_cor = df.iloc[:, -2:]
    df_cor.to_csv(path_to_data + '/train_cor.csv', index=False)

    FILE_NAME_TRAIN_RSS = path_to_data + '/train_rssi.csv'
    FILE_NAME_TRAIN_COORDS = path_to_data + '/train_cor.csv'
    # read training data
    X_train = genfromtxt(FILE_NAME_TRAIN_RSS, delimiter=',')
    y_train = genfromtxt(FILE_NAME_TRAIN_COORDS, delimiter=',')
    X_train[X_train == -100] = np.nan

    # test data
    X_tests = []
    y_tests = []
    FILE_NAME_TESTs = ['test.csv']
    models = ['AlexNet', 'VGG', 'ResNet']
    attack_types = ['FGSM', 'CW', 'L0', 'L2', 'Linf']
    for model in models:
        for attack_type in attack_types:
            FILE_NAME_TESTs.append('test_' + attack_type + '_' + model + '.csv')

    for FILE_NAME_TEST in FILE_NAME_TESTs:
    # 读取原始CSV文件
        df = pd.read_csv(path_to_data + '/test/' + FILE_NAME_TEST)

        # 选择前182列，保存到test_rssi.csv
        df_rssi = df.iloc[:, :182]
        df_rssi.to_csv(path_to_data + '/temp_rssi.csv', index=False)

        # 选择最后两列，保存到test_cor.csv
        df_cor = df.iloc[:, -2:]
        df_cor.to_csv(path_to_data + '/temp_cor.csv', index=False)
        FILE_NAME_TEST_RSS = path_to_data + '/temp_rssi.csv'
        FILE_NAME_TEST_COORDS = path_to_data + '/temp_cor.csv'
        # read test data
        X_test = genfromtxt(FILE_NAME_TEST_RSS, delimiter=',')
        y_test = genfromtxt(FILE_NAME_TEST_COORDS, delimiter=',')
        X_test[X_test == -100] = np.nan
        X_tests.append(X_test)
        y_tests.append(y_test)
    return (X_train, y_train, X_tests, y_tests, FILE_NAME_TESTs)


def distance(a, b):
    return np.sqrt(np.sum(np.power(a - b, 2)))


def bdist(a, b, sigma, eps, th, lth=-85, div=10):
    diff = a - b

    proba = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp( \
        -np.power(diff, 2) / (2.0 * sigma ** 2))

    proba[np.isnan(proba)] = eps
    proba[proba < th] = eps
    proba = np.log(proba)
    if a.ndim == 2:
        cost = np.sum(proba, axis=1)
    else:
        cost = np.sum(proba)

    inv = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        aa = np.logical_and(~np.isnan(a[i]), np.isnan(b))
        bb = np.logical_and(np.isnan(a[i]), ~np.isnan(b))

        nfound = np.concatenate((a[i, aa], b[bb]))
        for v in nfound[nfound > lth]:
            inv[i] += v - lth

    inv /= div
    cost -= inv

    return cost


def cluster_subset_kmeans(clusters, labels, pos, X_test):
    d = []
    for i, c in enumerate(kmeans.cluster_centers_):
        d.append(distance(pos[:2], c[:2]))

    center = np.argmin(d)

    return (ss[center], cs[center])


def cluster_subset_affinityprop(clusters, labels, X_test):
    subset = np.zeros(labels.shape[0]).astype(np.bool_)

    d = bdist(clusters, X_test, 5, 1e-3, 1e-25)
    idx = np.argsort(d)[::-1]

    cused = 0
    for c in idx[:5]:
        subset = np.logical_or(subset, c == labels)
        cused += 1

    return (subset, cused)


def bayes_position(X_train, y_train, X_test, N, sigma, eps, th, lth, div, y_test):
    diff = X_train - X_test

    proba = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp( \
        -np.power(diff, 2) / (2.0 * sigma ** 2))

    proba[np.isnan(proba)] = eps
    proba[proba < th] = eps
    proba = np.log(proba)
    cost = np.sum(proba, axis=1)

    inv = np.zeros(X_train.shape[0])
    for i in range(X_train.shape[0]):
        a = np.logical_and(~np.isnan(X_train[i]), np.isnan(X_test))
        b = np.logical_and(np.isnan(X_train[i]), ~np.isnan(X_test))

        nfound = np.concatenate((X_train[i, a], X_test[b]))
        for v in nfound[nfound > lth]:
            inv[i] += v - lth

    inv /= div
    cost -= inv

    idx = np.argsort(cost)[::-1]

    bias = 3
    position = np.zeros(2)
    N = min(N, y_train.shape[0])
    for i in range(N):
        weight = N - i
        if i == 0:
            weight += bias

        position += weight * y_train[idx[i]]

    position /= N * (N + 1) / 2 + bias

    return (np.array(position), np.mean(inv[idx[:20]]))


def position_route(method, X_train, y_train, X_test, y_test, clusters, labels,
                   N=5, sigma=5, eps=3e-4, th=1e-25, lth=-85, div=10):
    error = []
    error2D = []
    fdetect = 0
    y_pred = []
    cused = []

    for i in range(X_test.shape[0]):
        if i > 1:
            if method == 'km':
                subset, c = cluster_subset_kmeans(clusters, labels, pos, X_test[i])
                cused.append(c)
            elif method == 'ap':
                subset, c = cluster_subset_affinityprop(clusters, labels, X_test[i])
                cused.append(c)
        else:
            subset = np.ones(X_train.shape[0]).astype(np.bool_)

        if method == 'km':
            pos, q = bayes_position(X_train[subset], y_train[subset], X_test[i], N, sigma,
                                    eps, th, lth, div, y_test[i])

            if q > 50:
                pos, _ = bayes_position(X_train, y_train, X_test[i], N, sigma,
                                        eps, th, lth, div, y_test[i])
        elif method == 'ap':
            pos, _ = bayes_position(X_train[subset], y_train[subset], X_test[i], N, sigma,
                                    eps, th, lth, div, y_test[i])

        if i > 1:
            y_pred.append(pos)
            error.append(distance(y_test[i], y_pred[-1]))
            if np.isnan(error[-1]):
                distance(y_test[i], y_pred[-1])
            error2D.append(distance(y_test[i, 0:2], np.array(y_pred[-1])[0:2]))

    return (np.array(y_pred), np.array(error), np.array(error2D), fdetect, np.array(cused))



tsum = 0
# load data
X_train, y_train, X_tests, y_tests, test_names = load_data(path_to_database)
# prepare data for processing
ap_count = X_train.shape[1]
floors = [0]
X_ktrain = X_train.copy()
y_ktrain = y_train.copy()
X_aux = X_ktrain.copy()
X_aux[np.isnan(X_aux)] = 0
M = X_ktrain.shape[1]
corr = np.zeros((M, M))
cth = 500
keep = np.ones(M).astype(np.bool_)
for i in range(M):
    for j in range(i, M):
        if i != j:
            diff = np.abs(X_aux[:, i] - X_aux[:, j])
            corr[i, j] = corr[j, i] = np.sum(diff)
        else:
            corr[i, j] = cth
    if keep[i] and np.sum(corr[i, :] < cth) > 0:
        for p in np.where(corr[i, :] < cth)[0]:
            keep[p] = False
X_ktrain = X_ktrain[:, keep]
X_tests = [X_test[:, keep] for X_test in X_tests]
if method == 'km':
    C = 25
    kmeans = KMeans(n_clusters=C, n_init=500, n_jobs=2, tol=1e-9)
    labels = kmeans.fit_predict(y_ktrain)
    clusters = kmeans.cluster_centers_
    N = X_ktrain.shape[0]
    aux = np.zeros((C, C))
    for i in range(N):
        dist = np.zeros(N)
        for j in range(N):
            dist[j] = distance(y_ktrain[i], y_ktrain[j])
        idx = np.argsort(dist)
        for p in np.where(labels[idx] != labels[i])[0]:
            if dist[idx[p]] < 10:
                aux[labels[i], labels[idx[p]]] += 1
    ss = np.zeros((C, labels.size)).astype(np.bool)
    cs = np.zeros(C)
    rssl = []
    rssc = []
    for c in range(C):
        aux[c, c] = 1
        for i in np.where(aux[c] != 0)[0]:
            ss[c] = np.logical_or(ss[c], labels == i)
            cs[c] += 1
elif method == 'ap':
    N = X_ktrain.shape[0]
    affinity = np.zeros((N, N))
    for i in range(N):
        rate = (i + 1) / N
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain : {:^3.0f}%[{}->{}]".format(int(rate * 100), a, b),
              end="")
        affinity[i, :] = bdist(X_ktrain, X_ktrain[i], 5, 1e-3, 1e-25)
    cluster = AffinityPropagation(damping=0.5, affinity='precomputed')
    labels = cluster.fit_predict(affinity)
    C = np.unique(labels).size
    clusters = X_ktrain[cluster.cluster_centers_indices_]
else:
    print('Unknown method. Please choose either "km" or "ap".')
    quit()
base_error = None
for X_test, y_test, name in zip(X_tests, y_tests, test_names):
    t = time.perf_counter()
    # estimate positions for test data
    y, error3D, error2D, fdetect, cused = position_route(method, X_ktrain,
                                                         y_ktrain, X_test, y_test, clusters, labels, N=5, eps=1e-3)
    if name == 'test.csv':
        base_error = np.nan_to_num(error2D)
    tsum += time.perf_counter() - t
    print(name)
    print('Mean positioning error 2D: \t%.2lf m' % np.nanmean(error2D))
    error2D = np.nan_to_num(error2D)
    min_len = min(len(base_error), len(error2D))
    print('asr = {:.2}%'.format(np.count_nonzero(error2D[:min_len] >= base_error[:min_len] + 2.5) / min_len * 100))
    # if cused.size > 0:
    #    print('cused %.2lf' % np.mean(cused))