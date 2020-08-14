import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import date, timedelta
from tqdm import tqdm

train = pd.read_csv('Data/train.csv')
test = pd.read_csv('Data/test.csv')
train.columns = ['timestamp', 'year', 'month', 'day', 'hour', 'minute', 'second', 'temp_out', 'hum_out', 'air_out', 'hum_in', 'air_in', 'temp_in']
test.columns = ['timestamp', 'year', 'month', 'day', 'hour', 'minute', 'second', 'temp_out', 'hum_out', 'air_out', 'hum_in', 'air_in']
train['time'] = pd.to_datetime(train[['year', 'month', 'day', 'hour', 'minute']])
test['time'] = pd.to_datetime(test[['year', 'month', 'day', 'hour', 'minute']])

# interpolation
date = {'year': [], 'month': [], 'day': [], 'hour': [], 'minute': []}
for day in range(14, 32):
    for hour in range(0, 24):
        for minute in range(0, 60):
            date['year'].append(2019)
            date['month'].append(3)
            date['day'].append(day)
            date['hour'].append(hour)
            date['minute'].append(minute)
            
for day in range(1, 14):
    for hour in range(0, 24):
        for minute in range(0, 60):
            date['year'].append(2019)
            date['month'].append(4)
            date['day'].append(day)
            date['hour'].append(hour)
            date['minute'].append(minute)

full_date = pd.DataFrame(date)
full_date['time'] = pd.to_datetime(full_date[['year', 'month', 'day', 'hour', 'minute']])
full_train = full_date.loc[(full_date.time >= '2019-3-14 01') & (full_date.time < '2019-4-3 01'), :]

train = pd.merge(full_train, train, on=['year', 'month', 'day', 'hour', 'minute', 'time'], how='left')
train.drop_duplicates(['month', 'day', 'hour', 'minute'], inplace=True)
train.reset_index(inplace=True, drop=True)

# fill na
# train
def fill_na(row, col):
    if pd.isna(row[col]):
        # 先看看前一分钟有没有
        time = row.time - pd.Timedelta('1 minute')
        pre = train.loc[train.time == time]
        if len(pre) != 0 and pd.notna(pre[col].values[0]):
            return pre[col].values[0]
        else:
            # 再看昨天有没有
            time = row.time - pd.Timedelta('1 day')
            pre = train.loc[train.time == time]
            if len(pre) != 0 and pd.notna(pre[col].values[0]):
                return pre[col].values[0]
            else:
                # 再看看前天有没有
                time = row.time - pd.Timedelta('2 day')
                pre = train.loc[train.time == time]
                if len(pre) != 0 and pd.notna(pre[col].values[0]):
                    return pre[col].values[0]
                else:
                    # 否则直接找上一个非空值
                    pre = train.loc[(train.time < row.time) & (pd.notna(train[col]))]
                    return pre[col].values[0]
            
    return row[col]

train.drop_duplicates(['month', 'day', 'hour', 'minute'], inplace=True)

for feat in tqdm(['temp_out', 'hum_out', 'air_out', 'hum_in', 'air_in', 'temp_in']):
    train[feat] = train.apply(fill_na, axis=1, args=(feat,))

train['target'] = train['temp_in'] - train['temp_out']

# test
def avg_pre_next(row, col):
    if pd.isna(row[col]):
        pre_val = test.loc[test.time < row.time].iloc[-1][col]
        next_val = test.loc[test.time > row.time].iloc[0][col]
        return (pre_val + next_val) / 2
    return row[col]

for feat in tqdm(['temp_out', 'hum_out', 'air_out', 'hum_in', 'air_in']):
    test[feat] = test.apply(avg_pre_next, axis=1, args=(feat,))

# feature engineering
matrix = pd.concat([train, test], axis=0, ignore_index=True)

# 基本聚合特征
features = ['temp_out', 'hum_out', 'air_out', 'hum_in', 'air_in']
group_feats = []
for f in tqdm(features):
    matrix['MDH_{}_medi'.format(f)] = matrix.groupby(['month','day','hour'])[f].transform('median')
    matrix['MDH_{}_mean'.format(f)] = matrix.groupby(['month','day','hour'])[f].transform('mean')
    matrix['MDH_{}_max'.format(f)] = matrix.groupby(['month','day','hour'])[f].transform('max')
    matrix['MDH_{}_min'.format(f)] = matrix.groupby(['month','day','hour'])[f].transform('min')
    matrix['MDH_{}_std'.format(f)] = matrix.groupby(['month','day','hour'])[f].transform('std')

    group_feats.append('MDH_{}_medi'.format(f))
    group_feats.append('MDH_{}_mean'.format(f))

# 基本交叉特征
for f1 in tqdm(features + group_feats):
    for f2 in features + group_feats:
        if f1 != f2:
            colname = '{}_{}_ratio'.format(f1, f2)
            matrix[colname] = matrix[f1].values / matrix[f2].values

matrix = matrix.fillna(method='bfill')

# 历史信息提取
matrix['dt'] = matrix['day'].values + (matrix['month'].values - 3) * 31

features = features + ['temp_in']
for f in features:
    tmp_df = pd.DataFrame()
    for t in tqdm(range(15, 45)):
        tmp = matrix[matrix['dt'] < t].groupby(['hour'])[f].agg({'mean'}).reset_index()
        tmp.columns = ['hour', 'hit_{}_mean'.format(f)]
        tmp['dt'] = t
        tmp_df = tmp_df.append(tmp)
    
    matrix = matrix.merge(tmp_df, on=['dt', 'hour'], how='left')
    
matrix = matrix.fillna(method='bfill')

# lag features
features = ['temp_out', 'hum_out', 'air_out', 'hum_in', 'air_in']
lags = [1, 24]

for l in tqdm(lags):
    tmp = matrix[features+['time']].copy()
    tmp.time += pd.Timedelta(str(l) + ' hour')
    col_names = ['lag_'+str(l)+'_'+f for f in features]
    tmp.columns = col_names + ['time']
    matrix = pd.merge(matrix, tmp, on='time', how='left')
    del tmp

matrix.fillna(method='ffill', inplace=True)

# Trend
for feat in features:
    for n in lags:
        matrix['trend_'+str(n)+'_'+feat] = matrix[feat] - matrix['lag_'+str(n)+'_'+feat]
matrix.fillna(method='ffill', inplace=True)

# Bin features
features = ['temp_out', 'hum_out', 'air_out', 'hum_in', 'air_in']
for f in features:
    matrix[f+'_20_bin'] = pd.cut(matrix[f], 20, duplicates='drop').apply(lambda x:x.left).astype(int)
    matrix[f+'_50_bin'] = pd.cut(matrix[f], 50, duplicates='drop').apply(lambda x:x.left).astype(int)
    matrix[f+'_100_bin'] = pd.cut(matrix[f], 100, duplicates='drop').apply(lambda x:x.left).astype(int)
    matrix[f+'_200_bin'] = pd.cut(matrix[f], 200, duplicates='drop').apply(lambda x:x.left).astype(int)
    
features_20_bin = [f + '_20_bin' for f in features]
for f1 in tqdm(features_20_bin):
    for f2 in features:
        matrix['{}_{}_medi'.format(f1,f2)] = matrix.groupby([f1])[f2].transform('median')
        matrix['{}_{}_mean'.format(f1,f2)] = matrix.groupby([f1])[f2].transform('mean')
        matrix['{}_{}_max'.format(f1,f2)] = matrix.groupby([f1])[f2].transform('max')
        matrix['{}_{}_min'.format(f1,f2)] = matrix.groupby([f1])[f2].transform('min')

features_50_bin = [f + '_50_bin' for f in features]
for f1 in tqdm(features_50_bin):
    for f2 in features:
        matrix['{}_{}_medi'.format(f1,f2)] = matrix.groupby([f1])[f2].transform('median')
        matrix['{}_{}_mean'.format(f1,f2)] = matrix.groupby([f1])[f2].transform('mean')
        matrix['{}_{}_max'.format(f1,f2)] = matrix.groupby([f1])[f2].transform('max')
        matrix['{}_{}_min'.format(f1,f2)] = matrix.groupby([f1])[f2].transform('min')

features_100_bin = [f + '_100_bin' for f in features]
for f1 in tqdm(features_100_bin):
    for f2 in features:
        matrix['{}_{}_medi'.format(f1,f2)] = matrix.groupby([f1])[f2].transform('median')
        matrix['{}_{}_mean'.format(f1,f2)] = matrix.groupby([f1])[f2].transform('mean')
        matrix['{}_{}_max'.format(f1,f2)] = matrix.groupby([f1])[f2].transform('max')
        matrix['{}_{}_min'.format(f1,f2)] = matrix.groupby([f1])[f2].transform('min')

features_200_bin = [f + '_200_bin' for f in features]
for f1 in tqdm(features_200_bin):
    for f2 in features:
        matrix['{}_{}_medi'.format(f1,f2)] = matrix.groupby([f1])[f2].transform('median')
        matrix['{}_{}_mean'.format(f1,f2)] = matrix.groupby([f1])[f2].transform('mean')
        matrix['{}_{}_max'.format(f1,f2)] = matrix.groupby([f1])[f2].transform('max')
        matrix['{}_{}_min'.format(f1,f2)] = matrix.groupby([f1])[f2].transform('min')

# Rolling features
features = ['temp_out', 'hum_out', 'air_out', 'hum_in', 'air_in']
lags = [1,3,6,12,24]
matrix.set_index('time', inplace=True)

for feat in tqdm(features):
    for l in lags:
        lag_hour = str(l) + 'h'
        matrix['mean_'+str(l)+'_hours_'+feat] = matrix[feat].rolling(lag_hour).mean()
        matrix['median_'+str(l)+'_hours_'+feat] = matrix[feat].rolling(lag_hour).median()
        matrix['max_'+str(l)+'_hours_'+feat] = matrix[feat].rolling(lag_hour).max()
        matrix['min_'+str(l)+'_hours_'+feat] = matrix[feat].rolling(lag_hour).min()
        matrix['std_'+str(l)+'_hours_'+feat] = matrix[feat].rolling(lag_hour).std()
        matrix['skew_'+str(l)+'_hours_'+feat] = matrix[feat].rolling(lag_hour).skew()
        matrix['q1_'+str(l)+'_hours_'+feat] = matrix[feat].rolling(lag_hour).quantile(quantile=0.25)
        matrix['q3_'+str(l)+'_hours_'+feat] = matrix[feat].rolling(lag_hour).median(quantile=0.75)
        matrix['var_'+str(l)+'_hours_'+feat] = matrix['std_'+str(l)+'_hours_'+feat] / matrix['mean_'+str(l)+'_hours_'+feat]
        
matrix.reset_index(inplace=True)

# Save data
matrix.to_pickle('data.pkl')

# Modeling
data = pd.read_pickle('data.pkl')

features_to_drop = ['timestamp', 'year', 'second', 'time', 'target', 'temp_in']

num = int(len(train)*0.8)
X_train = data.iloc[:num].drop(features_to_drop, axis=1)
y_train = data.iloc[:num]['target']
# y_train = data.iloc[:num]['temp_in'] - pred_train_lr[:num]
X_val = data.iloc[num:len(train)].drop(features_to_drop, axis=1)
y_val = data.iloc[num:len(train)]['target']
# y_val = data.iloc[num:len(train)]['temp_in'] - pred_train_lr[num:len(train)]
X_test = data.loc[data.time >= '2019-4-3 01'].drop(features_to_drop, axis=1)

# Xgboost
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import plot_importance

model = XGBRegressor(max_depth=8,
                     n_estimators=50000,
                     min_child_weight=5, 
                     colsample_bytree=0.5, 
                     subsample=0.5, 
                     eta=0.001,
                     seed=2020, 
                     n_jobs=2)
model.fit(X_train, 
          y_train, 
          eval_metric='rmse', 
          eval_set=[(X_train, y_train), (X_val, y_val)], 
          verbose=500, 
          early_stopping_rounds=1000)

pred_test_xgb = model.predict(X_test, ntree_limit=model.best_ntree_limit)
submission = pd.DataFrame({'time': test.timestamp, 
                           'temperature': test.temp_out + pred_test_xgb})
submission.to_csv('submissions/xgb_submission.csv', index=False)