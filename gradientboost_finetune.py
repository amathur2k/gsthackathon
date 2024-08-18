import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the data
X_train = pd.read_csv('X_Train_Data_Input.csv', encoding='ascii')
Y_train = pd.read_csv('Y_Train_Data_Target.csv', encoding='ascii')

# Merge the datasets on 'ID'
data = pd.merge(X_train, Y_train, on='ID')

# Separate features and target
X = data.drop(columns=['ID', 'target'])
y = data['target']

# Handle missing values by imputing with the mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the parameter grid for Gradient Boosting
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Initialize the GridSearchCV
grid_search = GridSearchCV(estimator=GradientBoostingClassifier(random_state=42),
                           param_grid=param_grid,
                           scoring='recall',
                           cv=3,
                           n_jobs=-1)

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best model
best_gb_model = grid_search.best_estimator_

# Make predictions with the best model
best_gb_predictions = best_gb_model.predict(X_test)

# Evaluate the best model
best_gb_accuracy = accuracy_score(y_test, best_gb_predictions)
best_gb_report = classification_report(y_test, best_gb_predictions)

# Identify false negatives in the best model
best_gb_false_negatives = np.where((y_test == 1) & (best_gb_predictions == 0))[0]

print("Best Gradient Boosting Model Evaluation:")
print("Accuracy:", best_gb_accuracy)
print("\
Classification Report:")
print(best_gb_report)
print(f"\
Number of false negatives in best model: {len(best_gb_false_negatives)}")
print("Best Parameters:", grid_search.best_params_)
