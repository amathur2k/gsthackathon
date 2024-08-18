import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

# Load and preprocess data
x_train = pd.read_csv('X_Train_Data_Input.csv', encoding='ascii')
y_train = pd.read_csv('Y_Train_Data_Target.csv', encoding='ascii')
x_train.fillna(0, inplace=True)

class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.tensor(x_data.values, dtype=torch.float32)
        self.y_data = torch.tensor(y_data.values, dtype=torch.float32)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

class MassiveNN(nn.Module):
    def __init__(self):
        super(MassiveNN, self).__init__()
        self.fc1 = nn.Linear(22, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 4096)
        self.fc4 = nn.Linear(4096, 2048)
        self.fc5 = nn.Linear(2048, 1024)
        self.fc6 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.sigmoid(self.fc6(x))
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Prepare data
x_train_data = x_train.drop(columns=['ID'])
y_train_data = y_train['target']
x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train_data, y_train_data, test_size=0.2, random_state=42)

# Create datasets and loaders
train_dataset = CustomDataset(x_train_split, y_train_split)
val_dataset = CustomDataset(x_val_split, y_val_split)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Initialize model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MassiveNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f'Total number of trainable parameters: {count_parameters(model)}')

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    all_targets, all_predictions = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            predictions = (outputs > 0.5).float()
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    precision = precision_score(all_targets, all_predictions, pos_label=1)
    recall = recall_score(all_targets, all_predictions, pos_label=1)
    accuracy = accuracy_score(all_targets, all_predictions)

    print(f'Epoch {epoch+1}/{num_epochs}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}')