import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


# Load and merge datasets
x_train = pd.read_csv('X_Train_Data_Input.csv', encoding='ascii', nrows = 100)
y_train = pd.read_csv('Y_Train_Data_Target.csv', encoding='ascii', nrows = 100)
merged_df = pd.merge(x_train, y_train, on='ID')

# Impute missing values with mean
numeric_columns = merged_df.select_dtypes(include=['float64', 'int64']).columns
imputer = SimpleImputer(strategy='mean')
merged_df[numeric_columns] = imputer.fit_transform(merged_df[numeric_columns])

# Split the merged dataset into train and test sets
train_df, test_df = train_test_split(merged_df, test_size=0.2, random_state=42)

# Separate features and target
X_train = train_df.drop(['ID', 'target'], axis=1)
y_train = train_df['target']

X_full = merged_df.drop(['ID', 'target'], axis=1)
y_full = merged_df['target']


# Calculate class weights
class_weights = {0: 1, 1: 5}

# Train Random Forest model
rf_model = RandomForestClassifier(class_weight=class_weights, random_state=42)
rf_model.fit(X_full, y_full)
joblib.dump(rf_model, "random_forest.joblib")
#loaded_rf = joblib.load("random_forest.joblib")


xgb_model = xgb.XGBClassifier(scale_pos_weight=5, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_full, y_full)


# Train Gradient Boosting model
gb_model = GradientBoostingClassifier(random_state=42)
sample_weight = np.where(y_train == 1, 5, 1)
gb_model.fit(X_full, X_full, sample_weight=sample_weight)



