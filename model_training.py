
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the data
df = pd.read_csv('X_Train_Data_Input.csv', nrows = 5000)
y_train = pd.read_csv('Y_Train_Data_Target.csv', nrows = 5000)

# Merge the datasets on the 'ID' column
data = pd.merge(df, y_train, on='ID')

# Identify numeric columns
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
numeric_columns = [col for col in numeric_columns if col not in ['target']]

# Handle missing values for numeric columns only
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Separate features and target
X = data[numeric_columns]
y = data['target']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    #'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    #'SVM': SVC(kernel='rbf', gamma='scale', random_state=42,probability=True, class_weight={0: 1.0, 1: 5}),
    #'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    print(f"{name} Model:")
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
    print("")
