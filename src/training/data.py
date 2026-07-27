"""
Data loading, preprocessing, and dataset construction.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataPreparer:
    """Load raw CSV, split IC/OC data, and apply standardization."""

    def __init__(self, path, target_col=7):
        self.path = path
        self.target_col = target_col
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.raw = None
        self.labels = None

    def load(self):
        """Load CSV (no header), split features X, target y, label."""
        self.raw = pd.read_csv(self.path, header=None)
        self.labels = self.raw.iloc[:, -1].values.astype(int)
        feature_cols = list(range(self.raw.shape[1] - 1))
        X_all = self.raw.iloc[:, feature_cols].values.astype(np.float32)
        y_all = self.raw.iloc[:, self.target_col].values.astype(
            np.float32).reshape(-1, 1)
        return X_all, y_all, self.labels

    def split_and_scale(self, X_all, y_all, labels):
        """Split into IC train set and full test set, fit scaler on IC data."""
        ic_mask = labels == 1
        X_ic = X_all[ic_mask]
        y_ic = y_all[ic_mask]
        self.scaler_X.fit(X_ic)
        self.scaler_y.fit(y_ic)
        X_all_scaled = self.scaler_X.transform(X_all)
        y_all_scaled = self.scaler_y.transform(y_all)
        X_ic_scaled = X_all_scaled[ic_mask]
        y_ic_scaled = y_all_scaled[ic_mask]
        return (X_ic_scaled, y_ic_scaled), (X_all_scaled, y_all_scaled)


def create_dataset(X, y, window_size):
    """Convert 2D time series to LSTM input format [samples, time_steps, features]."""
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    n_samples = X_arr.shape[0] - window_size
    n_features = X_arr.shape[1]
    X_win = np.empty((n_samples, window_size, n_features), dtype=np.float32)
    y_win = np.empty((n_samples, 1), dtype=np.float32)
    for i in range(n_samples):
        X_win[i] = X_arr[i: i + window_size]
        y_win[i] = y_arr[i + window_size]
    return X_win, y_win
