
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
import os

# Check if running on Windows
if os.name == 'nt':
    print("Running on Windows")

# Load the data
x_train = pd.read_csv('X_Train_Data_Input.csv', encoding='ascii')
y_train = pd.read_csv('Y_Train_Data_Target.csv', encoding='ascii')

# Remove the 'ID' column from X as it's not a feature
X = x_train.drop('ID', axis=1)
y = y_train['target']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_val_imputed = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)

# Scale the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train.columns)
X_val_scaled = pd.DataFrame(scaler.transform(X_val_imputed), columns=X_val.columns)

# Select the top 10 most important features
important_features = ['Column18', 'Column1', 'Column2', 'Column17', 'Column4', 'Column3', 'Column7', 'Column8', 'Column19', 'Column5']
X_train_important = X_train_scaled[important_features]
X_val_important = X_val_scaled[important_features]

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Initialize the XGBoost model with GPU support
xgb_model = XGBClassifier(tree_method='gpu_hist', use_label_encoder=False, eval_metric='logloss')

# Perform grid search
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, n_jobs=-1, scoring='f1')
grid_search.fit(X_train_important, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions on the validation set
y_pred = best_model.predict(X_val_important)

# Calculate confusion matrix
cm = confusion_matrix(y_val, y_pred)

# Print results
print("Best Parameters:", grid_search.best_params_)
print("
Confusion Matrix:")
print(cm)
print("
Classification Report:")
print(classification_report(y_val, y_pred))

# Instructions for Windows users
print("
Ensure you have the following libraries installed:")
print("- pandas
- numpy
- scikit-learn
- xgboost")
print("Use the command 'pip install <library>' to install any missing libraries.")
