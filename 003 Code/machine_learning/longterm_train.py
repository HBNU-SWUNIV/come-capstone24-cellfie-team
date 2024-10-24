import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle as pkl
from tqdm import tqdm
from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

with open('./Input_merge_clim.pkl', 'rb') as f:
	X = pkl.load(f).to(device)
    
with open('./Output_merge_clim.pkl', 'rb') as f:
	Y = pkl.load(f).to(device)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

for inputs, targets in train_loader:
    batch_size, time_seq, num_attr = inputs.size()
    print(inputs.size())
    print(targets.size())
    break

for inputs, targets in test_loader:
#     batch_size, time_seq, num_attr = inputs.size()
    print(inputs.size())
    print(targets.size())
    break

class LSTMModel(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, num_linear_layers, linear_layer_dim, output_dim, lstm_layers=1, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.conv_out_channels = 64
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=self.conv_out_channels, kernel_size=1)

        self.layer_norm = nn.LayerNorm(self.conv_out_channels)
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, lstm_layers, batch_first=True, dropout=dropout if lstm_layers > 1 else 0)

        self.linear_layers = nn.ModuleList()
        for _ in range(num_linear_layers):
            self.linear_layers.append(nn.Linear(lstm_hidden_dim if len(self.linear_layers) == 0 else linear_layer_dim, linear_layer_dim))
            lstm_hidden_dim = linear_layer_dim

        self.output_layer = nn.Linear(linear_layer_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.mish = nn.Mish()

    def forward(self, x):
        # x = x.transpose(1, 2)  
        # x = self.conv1d(x)
        # x = x.transpose(1, 2) 

        # x = self.layer_norm(x)
        out, _ = self.lstm(x)
        out = self.mish(out[:, -1, :])  # Use the output of the last time step

        for layer in self.linear_layers:
            out = self.dropout(self.mish(layer(out)))

        out = self.output_layer(out)
        return out

def train_model(model, train_loader, criterion, optimizer, test_loader, num_epochs=100):
    model.train() 
    check_ii = 0
    for epoch in range(num_epochs):
        for inputs, targets in tqdm(train_loader):
            model.train() 
            optimizer.zero_grad()  
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()  
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        if (epoch+1) % 10 == 0 or epoch == 0:
            evaluate_model(model, test_loader, criterion)

def evaluate_model(model, test_loader, criterion):
    model.eval()  
    with torch.no_grad(): 
        losses = []
        for inputs, targets in tqdm(test_loader):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            
    avg_loss = sum(losses) / len(losses)
    print(f'Test Loss: {avg_loss:.4f}')
    test_mse_lst.append(avg_loss)

test_mse_lst = []

input_dim = X.size(-1); lstm_hidden_dim = 256
linear_layer_dim = 256; output_dim = 1
num_linear_layers=2; lstm_layers=16

model = LSTMModel(input_dim=input_dim, lstm_hidden_dim=lstm_hidden_dim, output_dim=output_dim, linear_layer_dim=linear_layer_dim,
                 num_linear_layers=num_linear_layers, lstm_layers=lstm_layers, dropout=0.2).to(device)
# model = nn.DataParallel(model).to('cuda')
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_model(model, train_loader, criterion, optimizer, test_loader, num_epochs=500)

evaluate_model(model, test_loader, criterion)

jit_model = torch.jit.script(model)
torch.jit.save(jit_model, './jit_model.pt')
print("Success Save")

with open("test_mse_lst.pkl","wb") as f:
    pkl.dump(test_mse_lst, f)