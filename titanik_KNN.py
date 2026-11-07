import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, RocCurveDisplay

from sklearn.model_selection import train_test_split


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

    for col in spend_cols:
        data[f'Log{col}'] = np.log1p(data[col])
    data['LogTotalSpend'] = np.log1p(data['TotalSpend'])

    cryo = data['CryoSleep'].astype('boolean')
    data.loc[cryo == True, spend_cols] = 0

    cryo_na = data['CryoSleep'].isna()
    has_spend = data[spend_cols].fillna(0).sum(axis=1) > 0

    data.loc[cryo_na & has_spend, 'CryoSleep'] = False

    data.loc[cryo_na & ~has_spend, 'CryoSleep'] = True

    for col in ['HomePlanet', 'Destination', 'Deck', 'Side']:
        data[col] = data[col].fillna('Unknown')




    data.drop(columns=['TotalSpend', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'])

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


"""
num_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc",  RobustScaler()),
])

cat_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore")),
])
"""
num_pipe = Pipeline([
    ("imp", KNNImputer(n_neighbors=5, weights='uniform')),
    ("sc",  RobustScaler()),
])

for col in cat_cols:
    data_train[col] = data_train[col].astype(object).fillna('U').astype(str)
    data_test[col] = data_test[col].astype(object).fillna('U').astype(str)


cat_pipe = Pipeline([
    ("imp", SimpleImputer(strategy='constant', fill_value='U')),

    ("ohe", OneHotEncoder(handle_unknown='ignore')),
])

preprocess = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])

pipe = Pipeline([
    ('preprocess', preprocess),
    ('select', SelectKBest(k = 30)),
    ('model', KNeighborsClassifier(n_jobs=-1)),
])
"""
param_grid = {
    'select__k': [10, 15, 20, 30, 'all'],
    'model__n_neighbors': [5, 10, 15, 20, 30, 51, 60, 80],
    'model__weights': ['uniform', 'distance'],
    'model__metric': ['manhattan', 'euclidean'],
    'model__p': [1, 2]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    pipe, param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    error_score='raise',
    return_train_score=True,
)

grid.fit(data_train, target_train)

print("\n" + "="*60)
print("Лучшие параметры:", grid.best_params_)
print(f"Лучший CV AUC: {grid.best_score_:.4f}")

"""
model = KNeighborsClassifier(n_neighbors = 30, weights = 'distance', metric = 'manhattan', p = 1)

data_train_train, data_train_val, target_train_train, target_train_val = train_test_split(data_train, target_train, test_size = 0.2, random_state = 42, stratify = target_train)

data_train_train_processed = preprocess.fit_transform(data_train_train)
data_train_val_processed = preprocess.transform(data_train_val)


data_train_processed = preprocess.fit_transform(data_train)

data_test_final = preprocess.transform(data_test)
model.fit(data_train_processed, target_train)


target_test = model.predict(data_test_final)


submission = pd.DataFrame({
    'PassengerId': data_sample_submission['PassengerId'],
    'Transported': target_test.astype(bool)
})

submission.to_csv('submission.csv', index=False)


