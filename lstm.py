import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import umap
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# # Load data
# X_train = pd.read_csv('X_train.csv')
# X_test = pd.read_csv('X_test.csv')
# y_train = pd.read_csv('y_train.csv')
# y_test = pd.read_csv('y_test.csv')

# # UMAP with 4 components
# umap_model = umap.UMAP(n_components=4, random_state=42)

# # Fit and transform the training set
# X_train_umap = umap_model.fit_transform(X_train)

# # Transform the test set
# X_test_umap = umap_model.transform(X_test)

# # Convert to DataFrame
# def normalize_features(df):
#     return (df - df.min()) / (df.max() - df.min())
# X_train_reduced = normalize_features(pd.DataFrame(X_train_umap, columns=[f'UMAP_{i+1}' for i in range(4)]))
# X_test_reduced = normalize_features(pd.DataFrame(X_test_umap, columns=[f'UMAP_{i+1}' for i in range(4)]))

# # Save or use the transformed datasets
# X_train_reduced.to_csv('X_train_reduced.csv', index=False)
# X_test_reduced.to_csv('X_test_reduced.csv', index=False)

# # Print shape to confirm
# print("Transformed X_train shape:", X_train_reduced.shape)
# print("Transformed X_test shape:", X_test_reduced.shape)

# ----------------------------------------------------------------------------------------------------------------

#Setup data for LSTM
X_train = pd.read_csv('X_train_reduced.csv').values
X_test = pd.read_csv('X_test_reduced.csv').values
y_train = pd.read_csv('y_train.csv').values.flatten()
y_test = pd.read_csv('y_test.csv').values.flatten()

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Custom Dataset class
class PestDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create dataset instances
dataset = PestDataset(X_train_tensor, y_train_tensor)
test_dataset = PestDataset(X_test_tensor, y_test_tensor)

# Split into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# LSTM Model
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x.unsqueeze(1)) 
        out = self.fc(lstm_out[:, -1, :])  
        return out.squeeze(-1)  

# Model initialization
input_size = X_train.shape[1]
model = LSTMPredictor(input_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with early stopping
epochs = 50
patience = 3  
best_r2 = float('-inf')
wait = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Validation phase
    model.eval()
    val_preds, val_trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
            val_preds.append(y_pred.cpu().numpy())
            val_trues.append(y_batch.cpu().numpy())
    
    val_preds = np.concatenate(val_preds)
    val_trues = np.concatenate(val_trues)
    val_r2 = r2_score(val_trues, val_preds)

    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}, Val R²: {val_r2:.6f}')
    
    # Early stopping check
    if val_r2 > best_r2:
        best_r2 = val_r2
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered!")
            break

# Final evaluation
model.eval()
y_preds, y_trues = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_pred = model(X_batch)
        y_preds.append(y_pred.cpu().numpy())
        y_trues.append(y_batch.cpu().numpy())

y_preds = np.concatenate(y_preds)
y_trues = np.concatenate(y_trues)

test_r2 = r2_score(y_trues, y_preds)
print(f'Final Test R² Score: {test_r2:.6f}')