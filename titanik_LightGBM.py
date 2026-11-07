from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import os
import kagglehub
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from catboost import CatBoostClassifier


def prepare(data):
    data = data.copy()
    data[['Group', 'MemberNum']] = data['PassengerId'].str.split('_', expand = True)
    data['Group'] = data['Group'].astype(int)
    data['MemberNum'] = data['MemberNum'].astype(int)

    data[['Deck', 'CabinNum', 'Side']] = data['Cabin'].str.split('/', expand = True)
    data['CabinNum'] = pd.to_numeric(data['CabinNum'], errors='coerce')

    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    data['TotalSpend'] = data[spend_cols].sum(axis=1, min_count=1)
    data['NoSpend'] = (data[spend_cols].fillna(0).sum(axis=1) == 0).astype(int)

    cryo = data['CryoSleep'].astype('boolean')
    data.loc[cryo == True, spend_cols] = 0

    cryo_na = data['CryoSleep'].isna()
    has_spend = data[spend_cols].fillna(0).sum(axis=1) > 0

    data.loc[cryo_na & has_spend, 'CryoSleep'] = False

    data.loc[cryo_na & ~has_spend, 'CryoSleep'] = True


    data['GroupSize'] = data.groupby('Group')['Group'].transform('count')
    data['IsAlone'] = (data['GroupSize'] == 1).astype(int)
    data['IsChild'] = (data['Age'] < 18).astype(int)

    data['Europa_Cryo'] = ((data['HomePlanet'] == 'Europa') & (data['CryoSleep'] == True)).astype(int)

    data['Earth_Cryo'] = ((data['HomePlanet'] == 'Earth') & (data['CryoSleep'] == True)).astype(int)

    data['Mars_Cryo'] = ((data['HomePlanet'] == 'Mars') & (data['CryoSleep'] == True)).astype(int)

    data['Europa_Cryo_f'] = ((data['HomePlanet'] == 'Europa') & (data['CryoSleep'] == False)).astype(int)

    data['Earth_Cryo_f'] = ((data['HomePlanet'] == 'Earth') & (data['CryoSleep'] == False)).astype(int)

    data['Mars_Cryo_f'] = ((data['HomePlanet'] == 'Mars') & (data['CryoSleep'] == False)).astype(int)
    data['3_high'] = data['NoSpend'] + data['Europa_Cryo'] + data['Mars_Cryo']
    data['3_low'] = data['Spa'] + data['RoomService'] + data['Earth_Cryo_f']

    return data.drop(columns=['PassengerId', 'Name', 'Cabin'])




base = './'
data_train= pd.read_csv(base + 'train.csv')
data_test = pd.read_csv(base + 'test.csv')
data_sample_submission = pd.read_csv(base + 'sample_submission.csv')

data_train = prepare(data_train)
data_test = prepare(data_test)


target_train = data_train['Transported']
data_train = data_train.drop(columns = ['Transported'])



num_cols = data_train.select_dtypes(include=np.number).columns.tolist()
cat_cols = data_train.select_dtypes(exclude=np.number).columns.tolist()



for col in cat_cols:

    data_train[col] = data_train[col].astype(object).fillna('NaN').astype(str)
    data_test[col] = data_test[col].astype(object).fillna('NaN').astype(str)


    le = LabelEncoder()
    all_values = pd.concat([data_train[col], data_test[col]], axis=0).astype(str)
    le.fit(all_values)

    data_train[col] = le.transform(data_train[col].astype(str))
    data_test[col] = le.transform(data_test[col].astype(str))

"""
n_train = len(data_train)
data_all = pd.concat([data_train, data_test], axis=0, ignore_index=True)

for col in cat_cols:
    data_all[col] = data_all[col].astype('category')

data_train = data_all.iloc[:n_train].reset_index(drop=True)
data_test = data_all.iloc[n_train:].reset_index(drop=True)
"""
cat_indices = [data_train.columns.get_loc(c) for c in cat_cols]


target_train = target_train.astype(int)


param_grid_1 = {
    'num_leaves': [15, 20, 27, 30, 31, 35],
    'min_child_samples': [5, 15, 20, 25, 30],
    'max_depth': [-1, 6, 12]

}

param_grid_2 = {
'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.1],
    'n_estimators': [100, 150, 200, 250, 300],
}

param_grid_3 = {
'subsample': [0.6, 0.7, 0.8, 1],
    'colsample_bytree': [0.5, 0.7, 0.8, 0.9, 1.0],
}

param_grid_4 = {
    'reg_alpha': [0.01, 0.05, 0.1, 0, 1, 10],
    'reg_lambda': [0, 0.1, 0.9, 1, 10],
}
param_grid = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.03, 0.05],
    'num_leaves': [15, 31],
    'max_depth': [-1, 5],
    'min_child_samples': [5, 10],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
}
"""
model = LGBMClassifier(
    random_state=42,
    verbose=-1,
    n_jobs=-1,
subsample_freq=1, colsample_bytree = 0.75, learning_rate = 0.13, max_depth = 4, min_child_samples = 18, n_estimators = 250, num_leaves = 13, subsample = 0.65, reg_alpha = 2.31, reg_lambda = 3.76
)
"""
"""
model1 = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=15,
    subsample=1,
    subsample_freq=1,
    colsample_bytree=1,
    reg_alpha=0.05,
    reg_lambda=0.01,
    random_state=42,
    verbose=-1,
    n_jobs=-1,
)

model2 = LGBMClassifier(
    n_estimators=250,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=5,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=1,
    reg_alpha=0.01,
    reg_lambda=1,
    random_state=42,
    verbose=-1,
    n_jobs=-1,
)



model3 = LGBMClassifier(
num_leaves=20,
    learning_rate=0.1,
    n_estimators=100,
    max_depth=-1,
    min_child_samples=20,
    subsample=1.0,
    subsample_freq=1,
    colsample_bytree=1.0,
    reg_alpha=0.05,
    reg_lambda=0.001,
    n_jobs=-1,
    verbosity=-1,
)
"""
"""
model = LGBMClassifier(
    random_state=42,
    verbose=-1,
    n_jobs=-1)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    model,
    param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=1,
    verbose=1,
    return_train_score=True,
)

grid.fit(
    data_train, target_train,
    categorical_feature=cat_indices,
)


print(grid.best_params_)
print(f"acc: {grid.best_score_:.4f}")

"""
model3 = LGBMClassifier(
    random_state=42,
    verbose=-1,
    n_jobs=-1,
    subsample_freq=1,
    colsample_bytree=0.8,
    learning_rate=0.03,
    max_depth=-1,
    min_child_samples=10,
    n_estimators=500,
    num_leaves=15,
    subsample=0.8,
)


model3.fit(
    data_train, target_train,
    categorical_feature=cat_indices,
)
target_test = model3.predict(data_test)


submission = pd.DataFrame({
    'PassengerId': data_sample_submission['PassengerId'],
    'Transported': target_test.astype(bool)
})

submission.to_csv('submission.csv', index=False)



"""
{'colsample_bytree': 0.8, 'learning_rate': 0.03, 'max_depth': 5, 'min_child_samples': 10, 'n_estimators': 500, 'num_leaves': 31, 'subsample': 0.8}
acc: 0.8135





model = LGBMClassifier(
    random_state=42,
    verbose=-1,
    n_jobs=-1,
    subsample_freq=1,
    colsample_bytree=0.8,
    learning_rate=0.05,
    max_depth=-1,
    min_child_samples=5,
    n_estimators=300,
    num_leaves=15,
    subsample=0.8,
    
    
    model = LGBMClassifier(
    random_state=42,
    verbose=-1,
    n_jobs=-1,
    subsample_freq=1,
    colsample_bytree=0.8,
    learning_rate=0.03,
    max_depth=-1,
    min_child_samples=10,
    n_estimators=500,
    num_leaves=15,
    subsample=0.8,
)
)
"""