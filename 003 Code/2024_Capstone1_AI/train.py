import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import time
import pickle as pkl
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device : {device}")

curr_path = './' # <-- Write your directory path

with open(curr_path+'/Input_merge_clim.pkl', 'rb') as f:
	X = pkl.load(f).to(device)
    
with open(curr_path+'/Output_merge_clim.pkl', 'rb') as f:
	Y = pkl.load(f).to(device)
	
# Reshaping the target data to apply batch size > 1
Y = Y.view(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 16
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

class GRUModel(nn.Module):
    def __init__(self, input_dim, gru_hidden_dim, num_linear_layers, linear_layer_dim, output_dim, gru_layers=1, dropout=0.5):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, gru_hidden_dim, gru_layers, batch_first=True, dropout=dropout if gru_layers > 1 else 0)

        self.linear_layers = nn.ModuleList()
        for _ in range(num_linear_layers):
            self.linear_layers.append(nn.Linear(gru_hidden_dim if len(self.linear_layers) == 0 else linear_layer_dim, linear_layer_dim))
            gru_hidden_dim = linear_layer_dim  # Update for the next layer if needed

        self.output_layer = nn.Linear(linear_layer_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Use the output of the last time step

        for layer in self.linear_layers:
            out = self.dropout(self.relu(layer(out)))

        out = self.output_layer(out)
        return out

def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    test_losses = []
    for epoch in range(num_epochs):
        for inputs, targets in tqdm(train_loader):
            model.train()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        test_loss = evaluate_model(model, test_loader, criterion)
        test_losses.append(test_loss) 
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}')

    return test_losses

def evaluate_model(model, test_loader, criterion):
    model.eval()
    with torch.no_grad():
        losses = []
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            
    avg_loss = sum(losses) / len(losses)
    return avg_loss 

input_dim = X.size(-1); gru_hidden_dim = 128
linear_layer_dim = 128; output_dim = 1
num_linear_layers=1; gru_layers=3

model = GRUModel(input_dim=input_dim, gru_hidden_dim=gru_hidden_dim, output_dim=output_dim, linear_layer_dim=linear_layer_dim,
                 num_linear_layers=num_linear_layers, gru_layers=gru_layers, dropout=0.3).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

test_losses = train_model(model, train_loader, criterion, optimizer, num_epochs=50)

model_path = curr_path+"/model.pt"
torch.save(model.state_dict(), model_path)

test_losses_path = curr_path+"/test_losses.pkl"
with open(test_losses_path, "wb") as f:
    pkl.dump(test_losses, f)

print("Success save Model and Loss")