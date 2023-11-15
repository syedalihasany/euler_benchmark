import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score



# Load input and output features from CSV files
input_features = pd.read_csv("input_features.csv")
output_features = pd.read_csv("output_features.csv")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(input_features.values, output_features.values, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

# Normalize the data using standard scaling
scaler_X = StandardScaler().fit(input_features.values)
scaler_y = StandardScaler().fit(output_features.values)

X_train_scaled = scaler_X.transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Convert the data to PyTorch tensors
# X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
# y_train = torch.tensor(y_train_scaled, dtype=torch.float32)
# X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
# y_test = torch.tensor(y_test_scaled, dtype=torch.float32)

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ... Rest of your code ...

# Convert the data to PyTorch tensors and move them to the GPU
X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)



# Create DataLoader for training and testing data
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64)

# Define the MLP model
class MLPModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        #self.fc2 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 1)

    
    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        x = self.tanh(x)
        x = self.fc4(x)
        return x
# Initialize the model
input_size = 7  # Assuming you have 6 input features
# Adjust the number of hidden units as needed
output_size = 1  # Assuming you have a single output feature

# model = MLPModel(input_size, output_size)

# Initialize the model and move it to the GPU
model = MLPModel(input_size, output_size).to(device)

# Define loss function for regression (Mean Squared Error)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Training loop
num_epochs = 1000  # Adjust the number of epochs as needed


# Training loop with batch iteration
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)  # Move inputs and targets to the GPU
        # Transfer data to the appropriate device (e.g., GPU if available)
        # inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Optionally print the mini-batch loss
        # print(f'Epoch [{epoch+1}/{num_epochs}], Batch {batch_idx+1}, Loss: {loss.item():.8f}')

    # Calculate average loss for the epoch
    epoch_loss = running_loss / len(train_loader)

    # Evaluate the model with the test set
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No gradients required for evaluation
        test_predictions = model(X_test)
        mse = nn.MSELoss()(test_predictions, y_test)

    # Print progress
    y_test_np = y_test.numpy()
    test_predictions_np = test_predictions.numpy()
    r_squared = r2_score(y_test_np, test_predictions_np)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.8f}, Mean Squared Error on Test Data: {mse.item():.8f}, R-squared on Test Data: {r_squared:.8f}')

print('Training finished.')



# Save the trained model
torch.save(model.state_dict(), "euler_model.pt")