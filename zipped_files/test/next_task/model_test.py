import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
#from euler_model import MLPModel 

# Load the new input features from a CSV file
new_input_features = pd.read_csv("input_features.csv")



# Performing scaling on the input data
input_scaler = StandardScaler()
input_scaler.fit(new_input_features)
X_new = input_scaler.transform(new_input_features)

# Convert the data to PyTorch tensors
X_new = torch.tensor(new_input_features.values, dtype=torch.float32)


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

# Load your trained model
model = MLPModel(input_size=7, output_size=1)  # Create an instance of your MLP model
model.load_state_dict(torch.load("euler_model.pt"))  # Load the saved model state

# Set the model to evaluation mode
model.eval()

# Make predictions on the new data
with torch.no_grad():
    new_predictions = model(X_new)

# You now have the predictions for the new data
# You can save or further analyze the predictions as needed

# Assuming you have new_output_features as ground truth for new data
# Load the new output features from a CSV file (if available)
new_output_features = pd.read_csv("output_features.csv")  # Replace with your new data file

# Performing scaling and normalization on the output features
output_scaler = StandardScaler()
output_scaler.fit(new_output_features)
X_new = output_scaler.transform(new_output_features)

# Calculate metrics on the new data (e.g., MSE and MAPE)
criterion = nn.MSELoss()
mse = criterion(new_predictions, torch.tensor(new_output_features.values, dtype=torch.float32))

# Calculate MAPE
abs_percentage_error = torch.abs((new_predictions - torch.tensor(new_output_features.values, dtype=torch.float32)) / torch.tensor(new_output_features.values, dtype=torch.float32))
mape = (abs_percentage_error.sum() / len(new_predictions)) * 100

print(f"Mean Squared Error on New Data: {mse.item():.8f}")
print(f"Mean Absolute Percentage Error (MAPE) on New Data: {mape:.8f}%")