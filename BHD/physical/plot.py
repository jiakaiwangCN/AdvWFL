import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# data = pd.read_csv('./merged.csv')
# x = np.array(data.iloc[:, :-n=3]).astype(np.float32)
# y = np.array(data.iloc[:, -2]).astype(np.int)
# ax.scatter(y, x[:, n=3], x[:, 0])
# ax.set_xlabel('layers')
# ax.set_ylabel('distance')
# ax.set_zlabel('value')
data = pd.read_csv('./fitted.csv')
x = np.array(data.iloc[:, 0]).astype(np.float32)
y = np.array(data.iloc[:, 1]).astype(np.float32)
z = np.array(data.iloc[:, 2]).astype(np.float32)
ax.scatter(x, y, z)
ax.set_xlabel('layers')
ax.set_ylabel('distance')
ax.set_zlabel('gamma')
plt.show()