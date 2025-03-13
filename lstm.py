import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score  # Import R² metric

# Load data
X_train = pd.read_csv('X_train.csv').values
X_test = pd.read_csv('X_test.csv').values
y_train = pd.read_csv('y_train.csv').values
y_test = pd.read_csv('y_test.csv').values

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).squeeze(-1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).squeeze(-1)

# Dataset class
class PestDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# DataLoader
train_dataset = PestDataset(X_train_tensor, y_train_tensor)
test_dataset = PestDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# LSTM Model
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x.unsqueeze(1))  
        out = self.fc(lstm_out[:, -1, :])  
        return out

# Model initialization
input_size = X_train.shape[1]
model = LSTMPredictor(input_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 14
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch).squeeze(-1)  # Ensure shape matches y_batch
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}')

# Evaluation
model.eval()
y_preds, y_trues = [], []

with torch.no_grad():
    test_loss = 0
    for X_batch, y_batch in test_loader:
        y_pred = model(X_batch).squeeze(-1)  # Ensure shape matches y_batch
        test_loss += criterion(y_pred, y_batch).item()
        y_preds.append(y_pred.cpu().numpy())  # Convert to NumPy
        y_trues.append(y_batch.cpu().numpy())

test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.6f}')

# Ensure correct concatenation
y_preds = np.concatenate(y_preds).flatten()  
y_trues = np.concatenate(y_trues).flatten()  

r2 = r2_score(y_trues, y_preds)
print(f'\n --- R2 Score on Test Set: {r2:.6f}')