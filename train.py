"""
Main training script: LSTM-Poisson residual + EWMA control chart (v3).
Runs the full pipeline: data loading, model training, EWMA control chart generation.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
DATA_PATH = r'd:\Project\python\ccfb\监控训练数据.csv'
OUTPUT_DIR = r'd:\Project\python\ccfb\results_v4'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Model hyperparameters
INPUT_DIM = 8
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2
WINDOW_SIZE = 14
EPOCHS = 200
LR = 0.001
PATIENCE = 15
EWMA_LAMBDAS = [0.01, 0.05, 0.1, 0.2]
TARGET_ARL0 = 200


# ============================================================
# Data Loading
# ============================================================
def load_and_preprocess(data_path):
    """Load CSV data and preprocess."""
    df = pd.read_csv(data_path, header=None)
    covariates = df.iloc[:, 0:6].values.astype(np.float32)
    exposure = df.iloc[:, 6].values.astype(np.float32)
    response = df.iloc[:, 7].values.astype(np.float32)
    label = df.iloc[:, 8].values.astype(int)

    scaler = StandardScaler()
    covariates_scaled = scaler.fit_transform(covariates).astype(np.float32)
    log_exposure = np.log(exposure + 1).astype(np.float32)

    print(f"Data shape: {df.shape}")
    print(
        f"Label distribution: {dict(zip(*np.unique(label, return_counts=True)))}")
    print(f"Response range: [{response.min():.1f}, {response.max():.1f}]")
    print(f"Exposure range: [{exposure.min():.1f}, {exposure.max():.1f}]")
    print(f"Zero count in response: {np.sum(response == 0)} (ZIP not needed)")

    return covariates_scaled, log_exposure, exposure, response, label, scaler


# ============================================================
# Dataset
# ============================================================
class SlidingWindowDataset(Dataset):
    """Sliding window dataset for LSTM."""

    def __init__(self, covariates, log_exposure, response, label, window_size=14):
        self.covariates = covariates
        self.log_exposure = log_exposure
        self.response = response
        self.label = label
        self.window_size = window_size
        self.valid_indices = list(range(window_size, len(covariates)))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        x_cov = self.covariates[t - self.window_size:t]
        x_exp = self.log_exposure[t - self.window_size:t]
        x_resp = self.response[t - self.window_size:t]
        x_input = np.concatenate(
            [x_cov, x_exp.reshape(-1, 1), x_resp.reshape(-1, 1)], axis=1)

        y = self.response[t]
        log_exp_t = self.log_exposure[t]
        lbl = self.label[t]

        return (
            torch.FloatTensor(x_input),
            torch.FloatTensor([y]),
            torch.FloatTensor([log_exp_t]),
            torch.IntTensor([lbl]),
        )


# ============================================================
# Model
# ============================================================
class LSTMPoissonModel(nn.Module):
    """LSTM model: output Poisson distribution mean parameter."""

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x, log_exposure):
        lstm_out, _ = self.lstm(x)
        h_last = lstm_out[:, -1, :]
        log_lambda = self.fc(h_last)
        mu = torch.exp(log_lambda + log_exposure)
        return mu, log_lambda


# ============================================================
# Loss Function
# ============================================================
def poisson_negative_log_likelihood(y, mu, eps=1e-8):
    """Poisson negative log-likelihood loss (with exposure offset)."""
    mu = torch.clamp(mu, eps, 1e6)
    nll = -(y * torch.log(mu + eps) - mu - torch.lgamma(y + 1))
    return nll.mean()


# ============================================================
# Residual Computation
# ============================================================
def compute_pearson_residuals(y, mu, eps=1e-8):
    """Pearson residuals: r = (y - mu) / sqrt(mu)."""
    mu_np = np.clip(mu.astype(np.float64), eps, 1e10)
    y_np = y.astype(np.float64)
    return (y_np - mu_np) / np.sqrt(mu_np)


def compute_deviance_residuals(y, mu, eps=1e-8):
    """Deviance residuals."""
    y_np = y.astype(np.float64)
    mu_np = np.clip(mu.astype(np.float64), eps, 1e10)

    deviance = np.zeros_like(y_np)
    for i in range(len(y_np)):
        yi, mui = y_np[i], mu_np[i]
        if yi == 0:
            term = 2 * mui
        else:
            term = 2 * (yi * np.log(yi / mui) - (yi - mui))
        deviance[i] = np.sign(yi - mui) * np.sqrt(np.abs(term))

    return deviance


def compute_standardized_residuals(y, mu, eps=1e-8):
    """Standardized residuals with rolling standardization."""
    pearson = compute_pearson_residuals(y, mu, eps)
    standardized = np.zeros_like(pearson)
    for i in range(len(pearson)):
        if i < 30:
            window = pearson[:i + 1]
        else:
            window = pearson[:i]
        if len(window) > 1:
            standardized[i] = (pearson[i] -
                               np.mean(window)) / (np.std(window) + eps)
        else:
            standardized[i] = pearson[i]
    return standardized


def compute_log_residuals(y, mu, eps=1e-8):
    """Log residuals: on logarithmic scale."""
    y_np = np.clip(y.astype(np.float64), eps, 1e10)
    mu_np = np.clip(mu.astype(np.float64), eps, 1e10)
    return np.log(y_np) - np.log(mu_np)


# ============================================================
# EWMA
# ============================================================
def compute_ewma(residuals, lam=0.05):
    """Compute EWMA statistic sequence."""
    z = np.zeros(len(residuals))
    z[0] = residuals[0]
    for t in range(1, len(residuals)):
        z[t] = (1 - lam) * z[t - 1] + lam * residuals[t]
    return z


def determine_control_limit_bilateral(ic_residuals, lam=0.05, target_arl0=400, n_sim=5000, seq_len=500):
    """Determine bilateral control limit h via Monte Carlo simulation."""
    print(f"  Determining control limit (lambda={lam}, target ARL0={target_arl0})...")

    ic_mean = np.mean(ic_residuals)
    ic_std = np.std(ic_residuals)
    ic_centered = ic_residuals - ic_mean

    def simulate_arl(h):
        rl_list = []
        for _ in range(n_sim):
            sample = np.random.choice(
                ic_centered, size=seq_len, replace=True) + ic_mean
            z = 0.0
            rl = seq_len + 1
            for t in range(seq_len):
                z = (1 - lam) * z + lam * sample[t]
                if abs(z - ic_mean) > h:
                    rl = t + 1
                    break
            rl_list.append(rl)
        return np.mean(rl_list)

    # Binary search
    h_low, h_high = 0.5 * ic_std, 10 * ic_std

    for _ in range(30):
        h_mid = (h_low + h_high) / 2
        arl = simulate_arl(h_mid)
        if arl < target_arl0:
            h_low = h_mid
        else:
            h_high = h_mid
        if abs(arl - target_arl0) / target_arl0 < 0.05:
            break

    final_arl = simulate_arl(h_mid)
    print(f"  Control limit h = {h_mid:.4f}, ARL0 = {final_arl:.1f}")
    return h_mid


# ============================================================
# Training
# ============================================================
def train_model(model, train_loader, val_loader, n_epochs=200, lr=0.001, patience=15):
    """Train LSTM-Poisson model."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x_batch, y_batch, exp_batch, _ in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            exp_batch = exp_batch.to(DEVICE)

            mu_pred, _ = model(x_batch, exp_batch)
            loss = poisson_negative_log_likelihood(y_batch, mu_pred)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for x_batch, y_batch, exp_batch, _ in val_loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                exp_batch = exp_batch.to(DEVICE)
                mu_pred, _ = model(x_batch, exp_batch)
                loss = poisson_negative_log_likelihood(y_batch, mu_pred)
                val_loss += loss.item()
                n_val += 1

        avg_val_loss = val_loss / max(n_val, 1)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.clone()
                          for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (epoch + 1) % 20 == 0:
            print(
                f"  Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if wait >= patience:
            print(f"  Early stopping at Epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses


# ============================================================
# Prediction & Residuals
# ============================================================
def predict_and_compute_residuals(model, dataset, residual_type='pearson'):
    """Predict on dataset and compute residuals."""
    model.eval()
    all_mu = []
    all_y = []
    all_label = []
    all_log_lambda = []

    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    with torch.no_grad():
        for x_batch, y_batch, exp_batch, lbl_batch in loader:
            x_batch = x_batch.to(DEVICE)
            exp_batch = exp_batch.to(DEVICE)

            mu_pred, log_lambda = model(x_batch, exp_batch)

            all_mu.append(mu_pred.cpu().numpy())
            all_y.append(y_batch.numpy())
            all_label.append(lbl_batch.numpy())
            all_log_lambda.append(log_lambda.cpu().numpy())

    mu_arr = np.concatenate(all_mu).flatten()
    y_arr = np.concatenate(all_y).flatten()
    label_arr = np.concatenate(all_label).flatten()
    log_lambda_arr = np.concatenate(all_log_lambda).flatten()

    if residual_type == 'pearson':
        residuals = compute_pearson_residuals(y_arr, mu_arr)
    elif residual_type == 'deviance':
        residuals = compute_deviance_residuals(y_arr, mu_arr)
    elif residual_type == 'standardized':
        residuals = compute_standardized_residuals(y_arr, mu_arr)
    elif residual_type == 'log':
        residuals = compute_log_residuals(y_arr, mu_arr)
    else:
        residuals = compute_pearson_residuals(y_arr, mu_arr)

    return y_arr, mu_arr, log_lambda_arr, residuals, label_arr


# ============================================================
# Plotting
# ============================================================
def plot_training_curve(train_losses, val_losses, output_path):
    """Plot training curve."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    plt.style.use('seaborn-v0_8-whitegrid')
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'SimSun']
    rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, 'k-', linewidth=1.2, label='Training Loss')
    ax.plot(val_losses, 'k--', linewidth=1.2, label='Validation Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Poisson Negative Log-Likelihood', fontsize=12)
    ax.set_title('(a) LSTM-Poisson Model Training Curve', fontsize=13)
    ax.legend(loc='upper right', frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_ic_ewma(ic_ewma, h, lam, ic_mean, output_path):
    """Plot IC EWMA control chart."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    plt.style.use('seaborn-v0_8-whitegrid')
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'SimSun']
    rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(10, 5))
    time_idx = np.arange(len(ic_ewma))
    ax.plot(time_idx, ic_ewma, 'k-', linewidth=0.7,
            alpha=0.8, label='EWMA statistic')
    ax.axhline(y=ic_mean + h, color='r', linestyle='--',
               linewidth=1.2, label=f'UCL = {ic_mean+h:.3f}')
    ax.axhline(y=ic_mean - h, color='r', linestyle='--',
               linewidth=1.2, label=f'LCL = {ic_mean-h:.3f}')
    ax.axhline(y=ic_mean, color='k', linestyle=':',
               linewidth=0.8, alpha=0.7, label='Center line')

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('EWMA Statistic', fontsize=12)
    ax.set_title(f'(a) Phase I: IC EWMA Control Chart (lambda={lam})', fontsize=13)
    ax.legend(loc='upper right', frameon=True, edgecolor='black', ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_full_ewma(all_ewma, all_labels, h, lam, ic_mean, output_path):
    """Plot full data EWMA control chart."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    plt.style.use('seaborn-v0_8-whitegrid')
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'SimSun']
    rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(12, 5))
    ic_mask = (all_labels == 1)
    oc_mask = (all_labels != 1)
    time_idx = np.arange(len(all_ewma))

    ax.plot(time_idx, all_ewma, 'gray', linewidth=0.4, alpha=0.5)

    ic_idx = np.where(ic_mask)[0]
    if len(ic_idx) > 0:
        ax.scatter(ic_idx, all_ewma[ic_mask],
                   c='black', s=3, alpha=0.3, label='IC', zorder=3)

    oc_idx = np.where(oc_mask)[0]
    if len(oc_idx) > 0:
        ax.scatter(oc_idx, all_ewma[oc_mask],
                   c='red', s=5, alpha=0.6, label='OC', zorder=3)

    ax.axhline(y=ic_mean + h, color='r', linestyle='--',
               linewidth=1.2, label=f'UCL = {ic_mean+h:.3f}')
    ax.axhline(y=ic_mean - h, color='r', linestyle='--',
               linewidth=1.2, label=f'LCL = {ic_mean-h:.3f}')
    ax.axhline(y=ic_mean, color='k', linestyle=':',
               linewidth=0.8, alpha=0.7, label='Center line')

    alarm_mask = np.abs(all_ewma - ic_mean) > h
    alarm_idx = np.where(alarm_mask)[0]
    if len(alarm_idx) > 0:
        ax.scatter(alarm_idx, all_ewma[alarm_mask], c='darkred',
                   marker='x', s=15, alpha=0.8, label='Alarm', zorder=4)

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('EWMA Statistic', fontsize=12)
    ax.set_title(
        f'(b) Phase II: Full Data EWMA Control Chart (lambda={lam})', fontsize=13)
    ax.legend(loc='upper right', frameon=True, edgecolor='black', ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_prediction_analysis(all_y, all_mu, all_log_lambda, all_residuals, all_labels, output_path):
    """Prediction analysis plot (2x2 subplots)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    plt.style.use('seaborn-v0_8-whitegrid')
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'SimSun']
    rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ic_m = (all_labels == 1)
    oc_m = (all_labels != 1)

    # (a) Observed vs Predicted
    ax = axes[0, 0]
    ax.plot(all_y, 'k-', linewidth=0.6, alpha=0.5, label='Observed y')
    ax.plot(all_mu, 'r--', linewidth=0.6, alpha=0.5, label='Predicted mu')
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('(a) Observed vs Predicted Poisson Mean', fontsize=12)
    ax.legend(loc='upper right', frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3)

    # (b) Log Poisson rate
    ax = axes[0, 1]
    ax.plot(all_log_lambda, 'k-', linewidth=0.6, alpha=0.7)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('log(lambda)', fontsize=11)
    ax.set_title('(b) Predicted Log Poisson Rate', fontsize=12)
    ax.grid(True, alpha=0.3)

    # (c) Residuals
    ax = axes[1, 0]
    ax.scatter(np.where(ic_m)[0], all_residuals[ic_m],
               c='black', s=3, alpha=0.3, label='IC')
    ax.scatter(np.where(oc_m)[0], all_residuals[oc_m],
               c='red', s=3, alpha=0.5, label='OC')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Pearson Residual', fontsize=11)
    ax.set_title('(c) Pearson Residuals (Black=IC, Red=OC)', fontsize=12)
    ax.legend(loc='upper right', frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3)

    # (d) Scatter
    ax = axes[1, 1]
    ax.scatter(all_mu[ic_m], all_y[ic_m], c='black', alpha=0.2, s=6, label='IC')
    ax.scatter(all_mu[oc_m], all_y[oc_m], c='red', alpha=0.3, s=6, label='OC')
    max_val = max(all_mu.max(), all_y.max())
    ax.plot([0, max_val], [0, max_val], 'k--',
            linewidth=1.0, alpha=0.5, label='Perfect prediction')
    ax.set_xlabel('Predicted mu', fontsize=11)
    ax.set_ylabel('Observed y', fontsize=11)
    ax.set_title('(d) Predicted vs Observed', fontsize=12)
    ax.legend(loc='upper left', frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_residual_distribution(all_residuals, all_labels, output_path):
    """Residual distribution plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    plt.style.use('seaborn-v0_8-whitegrid')
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'SimSun']
    rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ic_res = all_residuals[all_labels == 1]
    oc_res = all_residuals[all_labels != 1]

    ax = axes[0]
    ax.hist(ic_res, bins=50, alpha=0.5, color='lightgray',
            label='IC', density=True, edgecolor='black', linewidth=0.5)
    ax.hist(oc_res, bins=50, alpha=0.5, color='darkgray',
            label='OC', density=True, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Pearson Residual', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(a) IC vs OC Residual Distribution', fontsize=12)
    ax.legend(frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for idx, oc_type in enumerate([2, 3, 4, 5]):
        mask = (all_labels == oc_type)
        if mask.sum() > 0:
            ax.hist(all_residuals[mask], bins=30, alpha=0.4, color=colors[idx],
                    label=f'OC Type {oc_type}', density=True, edgecolor='black', linewidth=0.3)
    ax.hist(ic_res, bins=30, alpha=0.3, color='lightgray',
            label='IC', density=True, edgecolor='black', linewidth=0.3)
    ax.set_xlabel('Pearson Residual', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(b) Residual Distribution by Type', fontsize=12)
    ax.legend(frameon=True, edgecolor='black', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_oc_type_ewma(all_ewma, all_labels, h, lam, ic_mean, output_path):
    """Per-type EWMA control chart."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    plt.style.use('seaborn-v0_8-whitegrid')
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'SimSun']
    rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, oc_type in enumerate([2, 3, 4, 5]):
        ax = axes[idx // 2, idx % 2]

        type_mask = (all_labels == oc_type)
        ic_m = (all_labels == 1)
        time_idx = np.arange(len(all_ewma))

        ax.plot(time_idx, all_ewma, 'lightgray', linewidth=0.3, alpha=0.3)

        if ic_m.sum() > 0:
            ic_idx = np.where(ic_m)[0]
            ax.scatter(ic_idx, all_ewma[ic_m],
                       c='black', s=2, alpha=0.2, label='IC')

        if type_mask.sum() > 0:
            type_idx = np.where(type_mask)[0]
            ax.scatter(type_idx, all_ewma[type_mask],
                       c='red', s=6, alpha=0.7, label=f'OC Type {oc_type}')

        ax.axhline(y=ic_mean + h, color='r', linestyle='--',
                   linewidth=1.0, label=f'UCL={ic_mean+h:.2f}')
        ax.axhline(y=ic_mean - h, color='r', linestyle='--',
                   linewidth=1.0, label=f'LCL={ic_mean-h:.2f}')
        ax.axhline(y=ic_mean, color='k', linestyle=':',
                   linewidth=0.6, alpha=0.5)

        alarm_mask = (np.abs(all_ewma - ic_mean) > h) & type_mask
        if alarm_mask.sum() > 0:
            alarm_idx = np.where(alarm_mask)[0]
            ax.scatter(alarm_idx, all_ewma[alarm_mask],
                       c='darkred', marker='x', s=20, label='Alarm')

        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('EWMA', fontsize=10)
        ax.set_title(f'OC Type {oc_type} (n={type_mask.sum()})', fontsize=11)
        ax.legend(fontsize=8, frameon=True, edgecolor='black')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_arl_comparison(arl_results, results, ewma_lambdas, output_path):
    """ARL and detection rate comparison chart."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    plt.style.use('seaborn-v0_8-whitegrid')
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'SimSun']
    rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    oc_types = [2, 3, 4, 5]
    x = np.arange(len(oc_types))
    width = 0.2
    colors = ['#2c3e50', '#e74c3c', '#3498db', '#95a5a6']

    ax = axes[0]
    for i, lam in enumerate(ewma_lambdas):
        lam_key = str(float(lam))
        arl_vals = []
        for ot in oc_types:
            val = arl_results.get(lam_key, {}).get(f'OC_type{ot}_ARL', 0)
            arl_vals.append(val)
        ax.bar(x + i * width, arl_vals, width, label=f'lambda={lam}',
               color=colors[i], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('OC Type', fontsize=12)
    ax.set_ylabel('ARL_1', fontsize=12)
    ax.set_title('(a) Out-of-Control ARL Comparison', fontsize=13)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f'Type {t}' for t in oc_types])
    ax.legend(frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    for i, lam in enumerate(ewma_lambdas):
        lam_key = str(float(lam))
        det_rates = []
        for ot in oc_types:
            key = f'oc_type{ot}_alarm_rate'
            val = results.get(lam_key, {}).get(key, 0)
            det_rates.append(val)
        ax.bar(x + i * width, det_rates, width, label=f'lambda={lam}',
               color=colors[i], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('OC Type', fontsize=12)
    ax.set_ylabel('Detection Rate', fontsize=12)
    ax.set_title('(b) Detection Rate by OC Type', fontsize=13)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f'Type {t}' for t in oc_types])
    ax.legend(frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_residual_acf(all_residuals, all_labels, output_path, max_lags=50):
    """Residual autocorrelation analysis chart."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from scipy import stats
    plt.style.use('seaborn-v0_8-whitegrid')
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'SimSun']
    rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ic_res = all_residuals[all_labels == 1]

    # ACF
    ax = axes[0]
    n = len(ic_res)
    lags = range(1, min(max_lags, n // 2))
    acf_vals = []
    for lag in lags:
        if n - lag > 0:
            corr = np.corrcoef(ic_res[:-lag], ic_res[lag:])[0, 1]
            acf_vals.append(corr if not np.isnan(corr) else 0)
        else:
            acf_vals.append(0)

    ax.bar(lags, acf_vals, color='black', alpha=0.7, width=0.6)
    conf = 1.96 / np.sqrt(n)
    ax.axhline(y=conf, color='r', linestyle='--',
               linewidth=1.0, label='95% CI')
    ax.axhline(y=-conf, color='r', linestyle='--', linewidth=1.0)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Lag', fontsize=11)
    ax.set_ylabel('ACF', fontsize=11)
    ax.set_title('(a) Residual ACF (IC Data)', fontsize=12)
    ax.legend(frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3)

    # QQ plot
    ax = axes[1]
    ic_res_sorted = np.sort(ic_res)
    n_ic = len(ic_res_sorted)
    theoretical = stats.norm.ppf(
        (np.arange(1, n_ic + 1) - 0.5) / n_ic)
    ax.scatter(theoretical, ic_res_sorted, c='black', s=8, alpha=0.5)
    z = np.polyfit(theoretical, ic_res_sorted, 1)
    p = np.poly1d(z)
    ax.plot(theoretical, p(theoretical),
            'r--', linewidth=1.2, label='Reference line')
    ax.set_xlabel('Theoretical Quantiles', fontsize=11)
    ax.set_ylabel('Sample Quantiles', fontsize=11)
    ax.set_title('(b) Normal Q-Q Plot', fontsize=12)
    ax.legend(frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# Main Pipeline
# ============================================================
def main():
    print("=" * 60)
    print("LSTM-Poisson Residual + EWMA Control Chart v4.0")
    print("=" * 60)

    # --- Load data ---
    print("\n[1] Loading and preprocessing data...")
    covariates, log_exposure, exposure, response, label, scaler = load_and_preprocess(
        DATA_PATH)

    # --- Split IC and OC ---
    ic_mask = (label == 1)
    oc_mask = (label != 1)
    print(f"IC samples: {ic_mask.sum()}, OC samples: {oc_mask.sum()}")

    # --- Sliding window dataset ---
    print(f"\n[2] Building sliding window dataset (window={WINDOW_SIZE})...")

    ic_indices = np.where(ic_mask)[0]
    ic_start = ic_indices[0]
    ic_end = ic_indices[-1]

    ic_cov = covariates[ic_start:ic_end + 1]
    ic_log_exp = log_exposure[ic_start:ic_end + 1]
    ic_resp = response[ic_start:ic_end + 1]
    ic_lbl = label[ic_start:ic_end + 1]

    ic_dataset = SlidingWindowDataset(
        ic_cov, ic_log_exp, ic_resp, ic_lbl, window_size=WINDOW_SIZE)
    print(f"IC dataset size: {len(ic_dataset)}")

    # --- Train/val split ---
    train_size = int(0.8 * len(ic_dataset))
    val_size = len(ic_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        ic_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # --- Build model ---
    print(
        f"\n[3] Building LSTM-Poisson model (hidden={HIDDEN_DIM}, layers={NUM_LAYERS})...")
    model = LSTMPoissonModel(INPUT_DIM, HIDDEN_DIM,
                             NUM_LAYERS, DROPOUT).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # --- Train ---
    print("\n[4] Training model...")
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        n_epochs=EPOCHS, lr=LR, patience=PATIENCE
    )

    # --- Training curve ---
    plot_training_curve(train_losses, val_losses, os.path.join(
        OUTPUT_DIR, 'fig1_training_curve.png'))
    print("Training curve saved")

    # --- Phase I: IC residuals and control limits ---
    print("\n[5] Phase I: Computing IC residuals and determining control limits...")

    residual_types = ['pearson', 'deviance', 'standardized', 'log']
    best_residual_type = 'pearson'
    best_score = -1

    for rtype in residual_types:
        _, _, _, residuals_tmp, _ = predict_and_compute_residuals(
            model, ic_dataset, residual_type=rtype)
        ic_res_tmp = residuals_tmp

        ic_std = np.std(ic_res_tmp)
        ic_mean = np.mean(ic_res_tmp)

        full_dataset_tmp = SlidingWindowDataset(
            covariates, log_exposure, response, label, window_size=WINDOW_SIZE)
        _, _, _, all_res_tmp, all_lbl_tmp = predict_and_compute_residuals(
            model, full_dataset_tmp, residual_type=rtype)

        ic_res_all = all_res_tmp[all_lbl_tmp == 1]
        oc_res_all = all_res_tmp[all_lbl_tmp != 1]

        if len(oc_res_all) > 0:
            separation = abs(np.mean(ic_res_all) -
                             np.mean(oc_res_all)) / (np.std(ic_res_all) + 1e-8)
        else:
            separation = 0

        score = separation / (ic_std + 1e-8)
        print(f"  {rtype}: IC std={ic_std:.4f}, separation={separation:.4f}, score={score:.4f}")

        if score > best_score:
            best_score = score
            best_residual_type = rtype

    print(
        f"\n  Selected residual type: {best_residual_type} (score={best_score:.4f})")

    ic_y, ic_mu, ic_log_lambda, ic_residuals, ic_labels = predict_and_compute_residuals(
        model, ic_dataset, residual_type=best_residual_type
    )

    print(
        f"IC residual stats: mean={np.mean(ic_residuals):.4f}, std={np.std(ic_residuals):.4f}")
    print(
        f"IC predicted mu stats: mean={np.mean(ic_mu):.4f}, std={np.std(ic_mu):.4f}")

    # Control limits for different lambda values
    control_limits = {}

    for lam in EWMA_LAMBDAS:
        h = determine_control_limit_bilateral(
            ic_residuals, lam=lam, target_arl0=TARGET_ARL0, n_sim=5000, seq_len=800)
        control_limits[lam] = h

    # IC EWMA charts
    ic_mean_res = np.mean(ic_residuals)
    for lam in EWMA_LAMBDAS:
        ic_ewma = compute_ewma(ic_residuals, lam=lam)
        h = control_limits[lam]
        plot_ic_ewma(ic_ewma, h, lam, ic_mean_res,
                     os.path.join(OUTPUT_DIR, f'fig2_ic_ewma_lambda_{lam}.png'))

    print("IC EWMA control charts saved")

    # --- Phase II: Full data evaluation ---
    print("\n[6] Phase II: Evaluating on full data...")
    full_dataset = SlidingWindowDataset(
        covariates, log_exposure, response, label, window_size=WINDOW_SIZE
    )

    all_y, all_mu, all_log_lambda, all_residuals, all_labels = predict_and_compute_residuals(
        model, full_dataset, residual_type=best_residual_type
    )

    # Prediction analysis
    plot_prediction_analysis(all_y, all_mu, all_log_lambda, all_residuals, all_labels,
                             os.path.join(OUTPUT_DIR, 'fig3_prediction_analysis.png'))
    print("Prediction analysis saved")

    # Full EWMA charts
    for lam in EWMA_LAMBDAS:
        all_ewma = compute_ewma(all_residuals, lam=lam)
        h = control_limits[lam]
        plot_full_ewma(all_ewma, all_labels, h, lam, ic_mean_res,
                       os.path.join(OUTPUT_DIR, f'fig4_full_ewma_lambda_{lam}.png'))

    print("Full data EWMA control charts saved")

    # --- Performance metrics ---
    print("\n[7] Computing monitoring performance metrics...")

    results = {}
    for lam in EWMA_LAMBDAS:
        all_ewma = compute_ewma(all_residuals, lam=lam)
        h = control_limits[lam]

        ic_ewma_vals = all_ewma[all_labels == 1]
        oc_ewma_vals = all_ewma[all_labels != 1]

        ic_alarm_rate = np.mean(
            np.abs(ic_ewma_vals - ic_mean_res) > h) if len(ic_ewma_vals) > 0 else 0
        oc_alarm_rate = np.mean(
            np.abs(oc_ewma_vals - ic_mean_res) > h) if len(oc_ewma_vals) > 0 else 0

        results[str(float(lam))] = {
            'control_limit': float(h),
            'ic_alarm_rate': float(ic_alarm_rate),
            'oc_alarm_rate': float(oc_alarm_rate),
        }

        for oc_type in [2, 3, 4, 5]:
            oc_type_mask = (all_labels == oc_type)
            if oc_type_mask.sum() > 0:
                oc_type_ewma = all_ewma[oc_type_mask]
                alarm_rate = np.mean(np.abs(oc_type_ewma - ic_mean_res) > h)
                results[str(float(lam))
                        ][f'oc_type{oc_type}_alarm_rate'] = float(alarm_rate)
                results[str(float(lam))
                        ][f'oc_type{oc_type}_count'] = int(oc_type_mask.sum())

        print(f"\n  lambda={lam}:")
        print(f"    UCL/LCL = {ic_mean_res + h:.4f} / {ic_mean_res - h:.4f}")
        print(f"    IC false alarm rate = {ic_alarm_rate:.4f}")
        print(f"    OC detection rate = {oc_alarm_rate:.4f}")
        for oc_type in [2, 3, 4, 5]:
            key = f'oc_type{oc_type}_alarm_rate'
            if key in results[str(float(lam))]:
                print(
                    f"    OC Type {oc_type} detection rate = {results[str(float(lam))][key]:.4f} (n={results[str(float(lam))][f'oc_type{oc_type}_count']})")

    # --- ARL ---
    print("\n[8] Computing ARL...")

    arl_results = {}
    for lam in EWMA_LAMBDAS:
        all_ewma = compute_ewma(all_residuals, lam=lam)
        h = control_limits[lam]

        arl_results[str(float(lam))] = {}

        ic_ewma_full = all_ewma[all_labels == 1]
        if len(ic_ewma_full) > 0:
            ic_above = np.where(np.abs(ic_ewma_full - ic_mean_res) > h)[0]
            if len(ic_above) > 0:
                ic_arl = len(ic_ewma_full) / max(len(ic_above), 1)
            else:
                ic_arl = len(ic_ewma_full)
            arl_results[str(float(lam))]['IC_ARL'] = float(ic_arl)

        for oc_type in [2, 3, 4, 5]:
            oc_mask = (all_labels == oc_type)
            oc_ewma_vals = all_ewma[oc_mask]
            if len(oc_ewma_vals) > 0:
                first_alarm = np.where(
                    np.abs(oc_ewma_vals - ic_mean_res) > h)[0]
                if len(first_alarm) > 0:
                    oc_arl = np.mean(first_alarm + 1)
                else:
                    oc_arl = len(oc_ewma_vals)
                arl_results[str(float(lam))
                            ][f'OC_type{oc_type}_ARL'] = float(oc_arl)

        print(f"  lambda={lam}: {arl_results[str(float(lam))]}")

    # --- Plots ---
    print("\n[9] Plotting OC type EWMA control charts...")
    for lam in EWMA_LAMBDAS:
        all_ewma = compute_ewma(all_residuals, lam=lam)
        h = control_limits[lam]
        plot_oc_type_ewma(all_ewma, all_labels, h, lam, ic_mean_res,
                          os.path.join(OUTPUT_DIR, f'fig5_oc_type_ewma_lambda_{lam}.png'))

    print("OC type EWMA control charts saved")

    plot_residual_distribution(all_residuals, all_labels,
                               os.path.join(OUTPUT_DIR, 'fig6_residual_distribution.png'))
    print("Residual distribution saved")

    plot_arl_comparison(arl_results, results, EWMA_LAMBDAS,
                        os.path.join(OUTPUT_DIR, 'fig7_arl_comparison.png'))
    print("ARL comparison saved")

    plot_residual_acf(all_residuals, all_labels,
                      os.path.join(OUTPUT_DIR, 'fig8_residual_acf.png'))
    print("Residual ACF saved")

    # --- Save results ---
    print("\n[10] Saving experiment results...")

    summary = {
        'model_config': {
            'input_dim': INPUT_DIM,
            'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT,
            'window_size': WINDOW_SIZE,
            'loss': 'Poisson',
            'residual_type': best_residual_type,
        },
        'data_stats': {
            'total_samples': int(len(label)),
            'ic_samples': int(ic_mask.sum()),
            'oc_samples': int(oc_mask.sum()),
            'oc_type_counts': {int(k): int(v) for k, v in zip(*np.unique(label[label != 1], return_counts=True))},
        },
        'ewma_results': results,
        'arl_results': arl_results,
    }

    with open(os.path.join(OUTPUT_DIR, 'experiment_results.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'window_size': WINDOW_SIZE,
        'control_limits': {str(k): float(v) for k, v in control_limits.items()},
        'residual_type': best_residual_type,
    }, os.path.join(OUTPUT_DIR, 'lstm_poisson_model.pth'))

    print("\n" + "=" * 60)
    print("Experiment completed! Results saved to:", OUTPUT_DIR)
    print("=" * 60)

    return model, results, arl_results, control_limits


if __name__ == '__main__':
    model, results, arl_results, control_limits = main()
