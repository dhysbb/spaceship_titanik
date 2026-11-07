from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
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

    data_train[col] = data_train[col].astype('category')
    data_test[col] = data_test[col].astype('category')

for col in cat_cols:
    all_categories = data_train[col].cat.categories.union(
        data_test[col].cat.categories
    )
    data_train[col] = data_train[col].cat.set_categories(all_categories)
    data_test[col] = data_test[col].cat.set_categories(all_categories)


cat_indices = [data_train.columns.get_loc(c) for c in cat_cols]


target_train = target_train.astype(int)

param_grid = {
    'n_estimators': [300, 500],
    'learning_rate': [0.03, 0.05],
    'max_depth': [3, 5],
    'min_child_weight': [1, 5],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'gamma': [0, 0.1],
}

model = XGBClassifier(
    random_state=42,
    n_jobs=-1,
    eval_metric='auc',
    tree_method='hist',
    enable_categorical=True,
    colsample_bytree =  0.8,
    gamma = 0,
    learning_rate = 0.05,
    max_depth = 5,
    min_child_weight = 1,
    n_estimators = 300,
    subsample = 0.8
)
"""
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
    data_train, target_train
)

print(grid.best_params_)
print(f"acc: {grid.best_score_:.4f}")

"""
model.fit(
    data_train, target_train
)
target_test = model.predict(data_test)


submission = pd.DataFrame({
    'PassengerId': data_sample_submission['PassengerId'],
    'Transported': target_test.astype(bool)
})

submission.to_csv('submission.csv', index=False)
