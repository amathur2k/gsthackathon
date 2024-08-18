
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report
import os

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess data
x_train = pd.read_csv('X_Train_Data_Input.csv', encoding='ascii')
y_train = pd.read_csv('Y_Train_Data_Target.csv', encoding='ascii')

X = x_train.drop('ID', axis=1)
y = y_train['target']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_val_imputed = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train.columns)
X_val_scaled = pd.DataFrame(scaler.transform(X_val_imputed), columns=X_val.columns)

important_features = ['Column18', 'Column1', 'Column2', 'Column17', 'Column4', 'Column3', 'Column7', 'Column8', 'Column19', 'Column5']
X_train_important = X_train_scaled[important_features]
X_val_important = X_val_scaled[important_features]

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_important.values).to(device)
y_train_tensor = torch.LongTensor(y_train.values).to(device)
X_val_tensor = torch.FloatTensor(X_val_important.values).to(device)
y_val_tensor = torch.LongTensor(y_val.values).to(device)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Define the neural network model
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Function to train the model
def train_model(model, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Accuracy: {100 * correct / total:.2f}%')

# Hyperparameter tuning
input_size = len(important_features)
num_classes = 2
hidden_sizes = [32, 64, 128]
learning_rates = [0.001, 0.01, 0.1]
num_epochs = 50

best_accuracy = 0
best_model = None
best_params = {}

for hidden_size in hidden_sizes:
    for lr in learning_rates:
        model = Net(input_size, hidden_size, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        print(f"Training with hidden_size={hidden_size}, learning_rate={lr}")
        train_model(model, criterion, optimizer, num_epochs)
        
        # Evaluate the model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy:.2f}%')
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_params = {'hidden_size': hidden_size, 'learning_rate': lr}

print(f"Best Parameters: {best_params}")
print(f"Best Validation Accuracy: {best_accuracy:.2f}%")

# Evaluate the best model
best_model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = best_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(classification_report(y_true, y_pred))

# Save the best model
torch.save(best_model.state_dict(), 'best_model.pth')
print("Best model saved as 'best_model.pth'")


