import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, accuracy_score

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

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled.values).to(device)
y_train_tensor = torch.LongTensor(y_train.values).to(device)
X_val_tensor = torch.FloatTensor(X_val_scaled.values).to(device)
y_val_tensor = torch.LongTensor(y_val.values).to(device)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# Define the massive neural network model
class MassiveNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(MassiveNet, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        self.layers.append(nn.Dropout(0.3))

        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            self.layers.append(nn.Dropout(0.3))

        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], num_classes))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Set up model parameters for a massive model
input_size = X_train_scaled.shape[1]
hidden_sizes = [2048, 1024, 512, 256, 128, 64, 32, 16]  # Significantly increased number of layers and neurons
num_classes = 2

# Initialize the massive model
massive_model = MassiveNet(input_size, hidden_sizes, num_classes).to(device)

# Initialize the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(massive_model.parameters(), lr=0.001)


# Define the training function
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
        y_pred = []
        y_true = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(labels.cpu().numpy())

        # Calculate metrics
        precision = precision_score(y_true, y_pred, pos_label=1)
        recall = recall_score(y_true, y_pred, pos_label=1)
        accuracy = accuracy_score(y_true, y_pred)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Precision for 1s: {precision:.4f}, '
              f'Recall for 1s: {recall:.4f}, '
              f'Overall Accuracy: {accuracy:.4f}')

    return model


# Set the number of epochs
num_epochs = 100

# Train the model
trained_model = train_model(massive_model, criterion, optimizer, num_epochs)

# Save the model
torch.save(trained_model.state_dict(), 'massive_model.pth')
print("\nModel saved as 'massive_model.pth'")