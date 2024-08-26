import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

df = pd.read_csv('./fitted.csv')
df['target'] = 'gamma'
X = df.iloc[:, :2]
y = df.iloc[:, 2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)
params = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 5,
    'eta': 0.1,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'silent': True,
    'gpu_id': 0,
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor'
}

# 训练模型
num_rounds = 100
model = xgb.train(params, dtrain, num_rounds)
# 预测结果
y_pred = model.predict(dtest)

# 输出模型性能
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('MSE:', mse)
print('R^2:', r2)