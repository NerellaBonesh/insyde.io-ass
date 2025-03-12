import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
import os

# Define a custom neural network class with flexible dimensions
class LayoutPredictor(nn.Module):
    def __init__(self, feature_count=None, target_count=None):  # Custom initialization
        super(LayoutPredictor, self).__init__()
        if feature_count is None or target_count is None:
            raise ValueError("Feature count and target count must be specified")
        self.layer1 = nn.Linear(feature_count, 128)  # Reduced hidden layer size
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, target_count)
    
    def forward(self, x):
        x = torch.tanh(self.layer1(x))  # Using tanh activation instead of ReLU
        x = torch.tanh(self.layer2(x))
        x = self.layer3(x)
        return x

# Function to train the neural network
def train_network(X_training, y_training, X_validation, y_validation, iterations=100):
    feature_size = X_training.shape[1]
    target_size = y_training.shape[1]
    
    print(f"Setting up network with feature_size={feature_size}, target_size={target_size}")
    
    network = LayoutPredictor(feature_count=feature_size, target_count=target_size)
    loss_function = nn.MSELoss()
    optimizer = optim.RMSprop(network.parameters(), lr=0.0005)  # Using RMSprop instead of Adam
    
    X_training_tensor = torch.tensor(X_training, dtype=torch.float32)
    y_training_tensor = torch.tensor(y_training, dtype=torch.float32)
    X_validation_tensor = torch.tensor(X_validation, dtype=torch.float32)
    y_validation_tensor = torch.tensor(y_validation, dtype=torch.float32)
    
    for iteration in range(iterations):
        network.train()
        optimizer.zero_grad()
        predictions = network(X_training_tensor)
        loss = loss_function(predictions, y_training_tensor)
        loss.backward()
        optimizer.step()
        
        if (iteration + 1) % 10 == 0:
            network.eval()
            with torch.no_grad():
                train_outputs = network(X_training_tensor).numpy()
                val_outputs = network(X_validation_tensor).numpy()
                train_error = math.sqrt(mean_squared_error(y_training, train_outputs))
                val_error = math.sqrt(mean_squared_error(y_validation, val_outputs))
            print(f'Iteration [{iteration+1}/{iterations}], Loss: {loss.item():.4f}, Train Error: {train_error:.4f}, Validation Error: {val_error:.4f}')
    
    return network, train_outputs, val_outputs

if __name__ == "__main__":
    # Verify dataset availability
    data_file = "furniture_dataset.csv"
    if not os.path.exists(data_file):
        print(f"❌ Dataset `{data_file}` is missing!")
        exit(1)
    
    print("✅ Reading data from file...")
    data = pd.read_csv(data_file)
    features = np.array(data["features"].apply(eval).tolist())  # Changed from "input" to "features"
    targets = np.array(data["labels"].apply(eval).tolist())    # Changed from "output" to "labels"

    # Divide data into training and validation sets
    features_train, features_val, targets_train, targets_val = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )
    
    # Store validation data for later use
    val_data = pd.DataFrame({
        "features": [f.tolist() for f in features_val],
        "targets": [t.tolist() for t in targets_val]
    })
    val_data.to_csv("layout_validation_data.csv", index=False)
    print(f"✅ Validation data saved to `layout_validation_data.csv` with {len(features_val)} records.")

    # Initiate training process
    print("🚀 Starting network training...")
    model, train_results, val_results = train_network(features_train, targets_train, features_val, targets_val)
    
    # Persist the trained model
    model_save_file = "layout_model.pth"
    torch.save(model.state_dict(), model_save_file)
    print(f"✅ Model saved successfully to `{model_save_file}`.")