import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, RocCurveDisplay

from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer

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
num_pipe = Pipeline([
    ("imp", KNNImputer(n_neighbors=5)),
    ("sc",  RobustScaler()),
])

data_train['Deck'] = data_train['Deck'].fillna('U')
data_train['CabinNum'] = data_train['CabinNum'].fillna(-1)
data_train['Side'] = data_train['Side'].fillna('U')


cat_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),

    ("ohe", OneHotEncoder(handle_unknown='ignore')),
])

preprocess = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])


data_train_train, data_train_val, target_train_train, target_train_val = train_test_split(data_train, target_train, test_size = 0.2, random_state = 42, stratify = target_train)
data_train_train_processed = preprocess.fit_transform(data_train_train)
data_train_val_processed = preprocess.transform(data_train_val)
feature_names = preprocess.get_feature_names_out()
"""
param_grid = {
    'model__max_depth': [3, 5, 7, 10, 15, None],
    'model__min_samples_split': [2, 5, 10, 20],
    'model__min_samples_leaf': [1, 2, 5, 10],
    'model__criterion': ['gini', 'entropy'],
    'model__class_weight': [None, 'balanced'],
}
pipe = Pipeline([('preprocess', preprocess),
    ('model', DecisionTreeClassifier(random_state=42))])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid.fit(data_train, target_train)

print(f"Лучший depth: {grid.best_params_['model__max_depth']}")
print(f"Лучший min_sample_split: {grid.best_params_['model__min_samples_split']}")
print(f"Лучший 'model__min_samples_leaf': {grid.best_params_['model__min_samples_leaf']}")
print(f"Лучший 'model__criterion': {grid.best_params_['model__criterion']}")
print(f"Лучший 'model__class_weight': {grid.best_params_['model__class_weight']}")
print(f"acc: {grid.best_score_:.4f}")

cv_results = cross_validate(
    grid.best_estimator_, data_train, target_train,
    cv=cv, scoring='accuracy',
    return_train_score=True, n_jobs=-1
)

print(f"train auc: {cv_results['train_score'].mean():.4f}")
print(f"vl acc:   {cv_results['test_score'].mean():.4f}")
print(f"   {cv_results['train_score'].mean() - cv_results['test_score'].mean():.4f}")

"""
data_train_processed = preprocess.fit_transform(data_train)

model = DecisionTreeClassifier(
    criterion='gini',
    splitter='best',
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=10,
    random_state=42,
    class_weight=None
)


model.fit(data_train_processed, target_train)
data_test_final = preprocess.transform(data_test)

target_test = model.predict(data_test_final)


submission = pd.DataFrame({
    'PassengerId': data_sample_submission['PassengerId'],
    'Transported': target_test.astype(bool)
})

submission.to_csv('submission.csv', index=False)
