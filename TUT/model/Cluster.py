import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class Cluster:
    def __init__(self, file_path, min_clusters=2, max_clusters=5):
        self.file_path = file_path
        self.min_clusters = min_clusters
        self.max_cluster = max_clusters + 1
        file = pd.read_csv(file_path)
        self.data = np.array(file.iloc[:, :-3])
        self.site = np.array(file.iloc[:, -3:])

    def kmeans(self):
        best_label = None
        best_silhouette_avg = 0
        best_k = 0
        for k in range(self.min_clusters, self.max_cluster):
            # 创建KMeans模型
            kmeans = KMeans(n_clusters=k)

            # 拟合模型并进行预测
            labels = kmeans.fit_predict(self.data)

            # 打印每个样本的聚类标签和聚类中心
            silhouette_avg = silhouette_score(self.data, labels)
            if best_silhouette_avg < silhouette_avg:
                best_silhouette_avg = silhouette_avg
                best_label = labels
                best_k = k

        array = np.concatenate((self.data, self.site, best_label.reshape(-1, 1)), axis=1)
        last_column_values = array[:, -1]
        sorted_array = array[last_column_values.argsort()]
        df = pd.DataFrame(sorted_array, columns=[f'col_{i}' for i in range(1, 997)])
        groups = df.groupby(df.iloc[:, -1])
        k_arrays = [group.values for _, group in groups]

        array_name = [f'RSS{i}' for i in range(1, 993)] + ['x', 'y', 'z']
        new_file_names = []
        for i in range(len(k_arrays)):
            df = pd.DataFrame(k_arrays[i][:, :-1], columns=array_name)
            new_file_name = os.path.join(os.path.dirname(self.file_path), 'test_cluster' + str(i) + '.csv')
            df.to_csv(new_file_name, index=False)
            new_file_names.append(new_file_name)

        return best_k, new_file_names
