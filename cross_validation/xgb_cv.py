# Libraries
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier
import json

# Load data
X_train = pd.read_csv('../data/X_train.csv')
X_test = pd.read_csv('../data/X_test.csv')
y_train = pd.read_csv('../data/y_train.csv')
y_test = pd.read_csv('../data/y_test.csv')

# Convert y_train and y_test to series
y_train = y_train['readmitted']
y_test = y_test['readmitted']

# Remove or replace special characters in column names - xgboost cannot handle
X_train.columns = X_train.columns.str.replace(r'[{}\[\]":,]', '', regex=True)
X_test.columns = X_test.columns.str.replace(r'[{}\[\]":,]', '', regex=True)

# Calculate class imbalance ratio
scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

# Cross validation
param_grid = {'n_estimators': [100, 200, 500],
              'learning_rate': [0.05, 0.1],
              'max_depth': [4, 6, 8],
              'scale_pos_weight': [scale_pos_weight/2, scale_pos_weight, scale_pos_weight*1.5],
              'min_child_weight': [1, 5, 10],
              'subsample': [0.6, 0.8, 1.0],
              'colsample_bytree': [0.6, 0.8, 1.0]}
xgb = XGBClassifier(eval_metric = 'logloss',
                    random_state = 42)
cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
grid_search = RandomizedSearchCV(estimator = xgb,
                    param_distributions = param_grid,
                    scoring = 'f1',
                    n_iter = 50,
                    cv = cv)
grid_search.fit(X_train, y_train)

# Save best params and best f1 score to access after
results = {'best_params': grid_search.best_params_,
           'best_f1_score': grid_search.best_score_}
with open('/rds/general/project/hda_24-25/live/ML/Group14/Models/best_params_xgb.txt', 'w') as f:
    json.dump(results, f, indent = 4)
