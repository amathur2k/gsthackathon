
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

# Define the modified neural network model
class ModifiedNet(nn.Module):
    def __init__(self, input_size, num_layers, initial_neurons, num_classes):
        super(ModifiedNet, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, initial_neurons))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(initial_neurons))
        self.layers.append(nn.Dropout(0.3))
        
        # Hidden layers with geometric progression
        for i in range(1, num_layers):
            neurons = max(int(initial_neurons * (0.5 ** i)), 16)  # Ensure at least 16 neurons
            self.layers.append(nn.Linear(int(initial_neurons * (0.5 ** (i-1))), neurons))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(neurons))
            self.layers.append(nn.Dropout(0.3))
        
        # Output layer
        self.layers.append(nn.Linear(max(int(initial_neurons * (0.5 ** (num_layers-1))), 16), num_classes))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Set up model parameters
input_size = 22  # Assuming 22 features as in the original script
num_layers = 19
initial_neurons = 1024
num_classes = 2

# Initialize the model
modified_model = ModifiedNet(input_size, num_layers, initial_neurons, num_classes).to(device)

# Rest of the script (data loading, training function, etc.) remains the same
