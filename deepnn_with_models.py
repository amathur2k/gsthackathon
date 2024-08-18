import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, accuracy_score

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the first 100 rows of the training data
data = pd.read_csv('training_data.csv', nrows=800000)
print("Data loaded. Shape:", data.shape)

# Prepare the features and target
#X = data.drop(['ID', 'target','ColumnRFMod', 'ColumnXGBMod', 'ColumnGBMod', 'Column12rms'], axis=1)
X = data.drop(['ID', 'target'], axis=1)
y = data['target']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
y_train_tensor = torch.FloatTensor(y_train.values).to(device)
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
y_test_tensor = torch.FloatTensor(y_test.values).to(device)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# Define the neural network
class DeepNN(nn.Module):
    def __init__(self, input_size, hidden_layers, neurons_per_layer, neuron_reduction_ratio):
        super(DeepNN, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_size, neurons_per_layer))
        self.layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(hidden_layers - 1):
            neurons_per_layer = int(neurons_per_layer / neuron_reduction_ratio)
            self.layers.append(nn.Linear(self.layers[-2].out_features, neurons_per_layer))
            self.layers.append(nn.ReLU())

        # Output layer
        self.layers.append(nn.Linear(self.layers[-2].out_features, 1))
        self.layers.append(nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Hyperparameters
input_size = X_train.shape[1]
hidden_layers = 6
starting_neurons = 256
neuron_reduction_ratio = 2
learning_rate = 0.001
epochs = 10
threshold = 0.4
# Class weights
class_weights = torch.tensor([0.5, 10]).to(device)

# Initialize the model
model = DeepNN(input_size, hidden_layers, starting_neurons, neuron_reduction_ratio).to(device)


num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params}. threshold: {threshold}. hidden_layers: {hidden_layers}. starting_neurons: {starting_neurons}. neuron_reduction_ratio: {neuron_reduction_ratio}. class_weights: {class_weights}")

# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



# Training loop
for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        weights = class_weights[batch_y.long()]
        loss = criterion(outputs, batch_y) * weights
        loss = loss.mean()
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).squeeze()
        y_pred_binary = (y_pred > threshold).float()
        #y_pred_binary = (y_pred + 0.1 * torch.tensor(X_test['SVM_Score'].values) > threshold).float()
        #SVM_Score = torch.tensor(X_test['SVM_Score'].values)
        #y_pred_binary = (y_pred + torch.where(SVM_Score > 0.4, 0.3, 0) > threshold).float()


        false_negatives = ((y_test_tensor == 1) & (y_pred_binary == 0)).sum().item()
        false_positives = ((y_test_tensor == 0) & (y_pred_binary == 1)).sum().item()

        fn = ((y_test_tensor == 1) & (y_pred_binary == 0))
        false_negatives_indices = torch.nonzero(fn).squeeze().cpu().numpy()
        false_negatives_features = X_test.iloc[false_negatives_indices]
        false_negatives_pred = y_pred.cpu().numpy()[false_negatives_indices]
        false_negatives_with_pred = false_negatives_features.copy()
        false_negatives_with_pred['predicted_value'] = false_negatives_pred

        false_negatives_with_pred.to_csv("false_negatives_features.csv", index=False)

        precision = precision_score(y_test_tensor.cpu(), y_pred_binary.cpu())
        recall = recall_score(y_test_tensor.cpu(), y_pred_binary.cpu())
        accuracy = accuracy_score(y_test_tensor.cpu(), y_pred_binary.cpu())

        print(
            f"Epoch {epoch + 1}/{epochs}, False Negatives: {false_negatives}, False Positives: {false_positives}, Precision for 1's: {precision:.4f}, Recall for 1's: {recall:.4f}, Overall Accuracy: {accuracy:.4f}, -----------------------------")

print("Training completed.")

# Save the model
torch.save(model.state_dict(), "deep_nn_model.pth")
print("Model saved as 'deep_nn_model.pth'")


# Function to make predictions
def predict(X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(scaler.transform(X)).to(device)
        predictions = model(X_tensor).squeeze()
        return (predictions > 0.5).float().cpu().numpy()


# Example usage
print("\nExample prediction:")
sample_data = X.iloc[:5]
predictions = predict(sample_data)
print("Predictions:", predictions)

print("\nScript execution completed.")