import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler  # Import MinMaxScaler



# Load input and output features from CSV files
input_features = pd.read_csv("csv_input_features.csv")  
output_features = pd.read_csv("csv_output_features.csv")

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the input features using the scaler
input_features = scaler.fit_transform(input_features)

# Convert values in scientific notation to regular floating-point numbers
#input_features = input_features.applymap(lambda x: float(x) if 'e' in str(x) else x)
#output_features = output_features.applymap(lambda x: float(x) if 'e' in str(x) else x)

# Check data types and convert to float32 if needed
#input_features = input_features.astype(np.float32)
#output_features = output_features.astype(np.float32)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(input_features, output_features.values, test_size=0.2, random_state=42)

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader for training and testing data
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64)

# Define the MLP model
class MLPModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 12)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(12, 2000)
        self.fc3 = nn.Linear(2000, 16)
        self.fc4 = nn.Linear(16, 1)

    
    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x
    
# Initialize the model
input_size = 9  # Assuming you have 6 input features
# Adjust the number of hidden units as needed
output_size = 1  # Assuming you have a single output feature

model = MLPModel(input_size, output_size)

# Define loss function for regression (Mean Squared Error)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10000  # Adjust the number of epochs as needed


for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    
    # Calculate loss
    loss = criterion(outputs, y_train)
    
    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.8f}')

print('Training finished.')

# Make predictions on the test data
with torch.no_grad():
    test_predictions = model(X_test)

# Calculate the Mean Squared Error
mse = nn.MSELoss()(test_predictions, y_test)

print(f"Mean Squared Error on Test Data: {mse.item():.8f}")

# Calculate the Mean Absolute Percentage Error (MAPE) for model predictions
model.eval()
with torch.no_grad():
    test_predictions = model(X_test)
    abs_percentage_error = torch.abs((test_predictions - y_test) / y_test)
    mape = (abs_percentage_error.sum() / len(test_predictions)) * 100  # Convert to percentage

print(f"Mean Absolute Percentage Error (MAPE) on Test Data: {mape:.2f}%")


# Save the trained model
torch.save(model.state_dict(), "euler_model_2.pt")