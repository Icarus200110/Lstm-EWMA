"""
LSTM model and training utilities.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMModel(nn.Module):
    """Simple regression LSTM: input window sequence, output next-step prediction."""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        pred = self.fc(last_step)
        return pred


class EarlyStopping:
    """Early stopping: stop training if loss doesn't improve for patience epochs."""

    def __init__(self, patience=15, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def train_lstm(X_train, y_train, input_size, hidden_size=64, num_layers=2,
               dropout=0.2, epochs=200, batch_size=32, lr=1e-3, patience=15):
    """Train LSTM regression model on IC data with EarlyStopping."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = LSTMModel(input_size, hidden_size, num_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    stopper = EarlyStopping(patience=patience)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(dataset)
        if epoch % 50 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>4d}/{epochs}  train_loss = {epoch_loss:.6f}")
        stopper.step(epoch_loss)
        if stopper.should_stop:
            print(
                f"    EarlyStopping at epoch {epoch}, best_loss = {stopper.best_loss:.6f}")
            break
    return model


def predict(model, X_data):
    """Predict using trained model, return numpy array."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_data, dtype=torch.float32).to(device)
        pred = model(X_t).cpu().numpy()
    return pred
