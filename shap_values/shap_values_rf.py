# Import libraries
import joblib
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

# Load model and X_test
X_train = pd.read_csv('../data/X_train.csv')
y_train = pd.read_csv('../data/y_train.csv')
X_test = pd.read_csv('../data/X_test.csv')
y_train = y_train['readmitted']

# Build model
rf = RandomForestClassifier(n_estimators = 500, 
                            max_depth = 50,
                            min_samples_split = 20,
                            min_samples_leaf = 10,
                            criterion = 'entropy',
                            max_features = 'log2',
                            class_weight = 'balanced',
                            random_state = 42)
rf.fit(X_train, y_train)

# Get SHAP values
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# Save SHAP values
joblib.dump(shap_values, '/rds/general/project/hda_24-25/live/TDS/hc724/Models/shap/shap_values_rf.pkl')
