import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

df = pd.read_csv('./fitted.csv')
layers = np.array(df.iloc[:, 0])
distance = np.array(df.iloc[:, 1])
gamma = np.array(df.iloc[:, 2])

x = np.column_stack((layers , distance))
poly_reg = PolynomialFeatures(degree=15, include_bias=False)
X_ploy = poly_reg.fit_transform(x)
regr = LinearRegression()
regr.fit(X_ploy, gamma)
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(regr.predict(X_ploy), gamma)
print("Mse: {}".format(mse))
print("R^2: {}".format(regr.score(X_ploy, gamma)))