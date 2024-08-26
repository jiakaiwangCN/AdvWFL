import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split


def get_data_as_numpy(file_name):
    data = pd.read_csv(file_name)
    x = np.array(data.iloc[:, :-1]).astype(np.float32)
    y = np.array(data.iloc[:, -1]).astype(np.int)
    return x, y


tree = tree.DecisionTreeClassifier(criterion='entropy')
feature, label = get_data_as_numpy('./merged.csv')
feature_train, feature_test, label_train, label_test = train_test_split(feature, label, test_size=0.2, shuffle=True)
tree = tree.fit(feature_train, label_train)
# 测试
accuracy = tree.score(feature_test, label_test)
print(accuracy)
