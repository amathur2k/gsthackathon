import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, accuracy_score
from joblib import dump

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

def impute_df(df, id_col='ID', strategy='mean'):
    df_no_id = df.drop(id_col, axis=1)
    imputed = pd.DataFrame(SimpleImputer(strategy=strategy).fit_transform(df_no_id), columns=df_no_id.columns)
    return pd.concat([df[id_col], imputed], axis=1)


# Load, merge, impute datasets and separate features and target
x = pd.read_csv('X_Train_Data_Input.csv', encoding='ascii', nrows = 800000)
y = pd.read_csv('Y_Train_Data_Target.csv', encoding='ascii', nrows = 800000)
train_xy_imputed = impute_df(pd.merge(x, y, on='ID'))
train_x_imputed = train_xy_imputed.drop(['ID', 'target'], axis=1)
train_y_imputed = train_xy_imputed['target']

train_small_xy_imputed, validation_xy_imputed = train_test_split(train_xy_imputed, test_size=0.2, random_state=42)
train_small_x_imputed, train_small_y_imputed = train_small_xy_imputed.drop(['target', 'ID'], axis=1), train_small_xy_imputed['target']
validation_x_imputed, validation_y_imputed = validation_xy_imputed.drop(['target', 'ID'], axis=1), validation_xy_imputed['target']

rf_model = RandomForestClassifier(class_weight={0: 1.0, 1: 5}, random_state=42, n_jobs=-1).fit(train_small_x_imputed, train_small_y_imputed)
xgb_model = xgb.XGBClassifier(scale_pos_weight=5, random_state=42, eval_metric='logloss').fit(train_small_x_imputed, train_small_y_imputed)

predictions_df = pd.DataFrame({
    'ID': train_xy_imputed['ID'],
    'RF_Score': rf_model.predict_proba(train_x_imputed)[:, 1],
    'XGB_Score': xgb_model.predict_proba(train_x_imputed)[:, 1]
})
final_df = pd.merge(train_xy_imputed, predictions_df, on='ID')
#final_df.to_csv('training_data.csv', index=False)
#print("Final dataframe saved to 'training_data.csv'")
#sys.exit(0)

#----------------------
#final_df = pd.read_csv('training_data.csv', nrows=800000)


train_small_xy_imputed, validation_xy_imputed = train_test_split(final_df, test_size=0.2, random_state=42, stratify=final_df['target'])
train_small_x_imputed, train_small_y_imputed = train_small_xy_imputed.drop(['target', 'ID'], axis=1), train_small_xy_imputed['target']
validation_x_imputed, validation_y_imputed = validation_xy_imputed.drop(['target', 'ID'], axis=1), validation_xy_imputed['target']
#----------------------

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

scaler = StandardScaler()
# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(scaler.fit_transform(train_small_x_imputed)).to(device)
y_train_tensor = torch.FloatTensor(train_small_y_imputed.values).to(device)
X_validation_tensor = torch.FloatTensor(scaler.transform(validation_x_imputed)).to(device)
y_validation_tensor = torch.FloatTensor(validation_y_imputed.values).to(device)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Hyperparameters
input_size = train_small_x_imputed.shape[1]
hidden_layers = 6
starting_neurons = 256
neuron_reduction_ratio = 2
learning_rate = 0.001
epochs = 2
threshold = 0.4
class_weights = torch.tensor([0.5, 10]).to(device)

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
        y_pred = model(X_validation_tensor).squeeze()
        y_pred_binary = (y_pred > threshold).float()

        false_negatives = ((y_validation_tensor == 1) & (y_pred_binary == 0)).sum().item()
        false_positives = ((y_validation_tensor == 0) & (y_pred_binary == 1)).sum().item()


        precision = precision_score(y_validation_tensor.cpu(), y_pred_binary.cpu())
        recall = recall_score(y_validation_tensor.cpu(), y_pred_binary.cpu())
        accuracy = accuracy_score(y_validation_tensor.cpu(), y_pred_binary.cpu())

        print(
            f"Epoch {epoch + 1}/{epochs}, False Negatives: {false_negatives}, False Positives: {false_positives}, Precision for 1's: {precision:.4f}, Recall for 1's: {recall:.4f}, Overall Accuracy: {accuracy:.4f}, -----------------------------")

print("Training completed.")

# Save the model
torch.save(model.state_dict(), "deep_nn_model.pth")
dump(rf_model, 'rf_model.joblib')
dump(xgb_model, 'xgb_model.joblib')