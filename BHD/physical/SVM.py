from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_data_as_numpy(file_name):
    data = pd.read_csv(file_name)
    x = np.array(data.iloc[:, :-1]).astype(np.float32)
    y = np.array(data.iloc[:, -1]).astype(np.int)
    return x, y


clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
get_data_as_numpy('./merged.csv')
feature, label = get_data_as_numpy('./merged.csv')
feature_train, feature_test, label_train, label_test = train_test_split(feature, label, test_size=0.2, shuffle=True)
clf = clf.fit(feature_train, label_train)
accuracy = clf.score(feature_test, label_test)
print(accuracy)
