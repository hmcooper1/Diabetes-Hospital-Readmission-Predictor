# Libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
import json

# Load data
X_train = pd.read_csv('../data/X_train.csv')
X_test = pd.read_csv('../data/X_test.csv')
y_train = pd.read_csv('../data/y_train.csv')
y_test = pd.read_csv('../data/y_test.csv')

# Convert y_train and y_test to series
y_train = y_train['readmitted']
y_test = y_test['readmitted']

# Cross validation
rf = RandomForestClassifier(random_state = 42,
                            class_weight = 'balanced')
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [10, 25, 50, None],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [5, 10, 20],
    'max_features': ['sqrt', 'log2'],
    'criterion': ['gini', 'entropy']
}
cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
grid_search = RandomizedSearchCV(estimator = rf,
                    param_distributions = param_grid,
                    scoring = 'f1',
                    n_iter = 50,
                    cv = cv)                           
grid_search.fit(X_train, y_train)
grid_search.best_params_
print("Best params:", grid_search.best_params_)
print("Best AUC from CV:", grid_search.best_score_)

# Save best params and best f1 score to access after
results = {'best_params': grid_search.best_params_,
           'best_f1_score': grid_search.best_score_}
with open('/rds/general/project/hda_24-25/live/ML/Group14/Models/best_params_rf.txt', 'w') as f:
    json.dump(results, f, indent = 4)
