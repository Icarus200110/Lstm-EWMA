"""
Main experiment script: LSTM-EWMA with three experiments.
Exp1: Baseline comparison (ARIMA-EWMA vs LSTM vs LSTM-EWMA)
Exp2: Sensitivity analysis (lambda, window size, ROC)
Exp3: Ablation study
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import OrderedDict
from statsmodels.tsa.arima.model import ARIMA

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# Configuration
# ============================================================
DATA_PATH = r"d:\Project\python\ccfb\监控训练数据.csv"
TARGET_COL = 7
WINDOW_SIZE = 10
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.2
EPOCHS = 200
BATCH_SIZE = 32
LR = 1e-3
PATIENCE = 15
EWMA_LAMBDA = 0.05
ARIMA_ORDER = (2, 1, 2)
SEED = 42


def set_seed(seed):
    """Fix random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        y_all = self.raw.iloc[:, self.target_col].values.astype(np.float32).reshape(-1, 1)
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


def compute_ewma(e_std, lam=0.05):
    """EWMA recursion: Z_t = lambda * e_std_t + (1 - lambda) * Z_{t-1}."""
    Z = np.zeros_like(e_std, dtype=np.float64)
    Z[0] = lam * e_std[0]
    for t in range(1, len(e_std)):
        Z[t] = lam * e_std[t] + (1 - lam) * Z[t - 1]
    return Z


def compute_control_limit(e_std_ic, lam=0.05, k=3):
    """Compute EWMA control limit width h based on IC phase residuals."""
    Z_ic = compute_ewma(e_std_ic, lam)
    h = k * np.std(Z_ic, ddof=1)
    return h


def forecast_arima_resid(fitted_model, y_full, ic_count, order=(2, 1, 2)):
    """
    Compute ARIMA residuals for the full dataset.
    Fit on IC data, then compute one-step-ahead residuals for all data.
    """
    y_ic = y_full[:ic_count]
    model_ic = ARIMA(y_ic, order=order)
    fitted_ic = model_ic.fit()

    residuals_full = np.full(len(y_full), np.nan, dtype=np.float64)
    fitted_ic_values = np.asarray(fitted_ic.fittedvalues)
    residuals_full[:ic_count] = y_ic[:len(
        fitted_ic_values)] - fitted_ic_values

    try:
        forecast_result = fitted_ic.apply(y_full)
        y_hat_full = np.asarray(forecast_result.fittedvalues)
        residuals_full = y_full - y_hat_full
    except Exception:
        for t in range(ic_count, len(y_full)):
            try:
                model_t = ARIMA(y_full[:t], order=order)
                fitted_t = model_t.fit()
                residuals_full[t] = y_full[t] - \
                    fitted_t.fittedvalues.values[-1]
            except Exception:
                residuals_full[t] = residuals_full[t - 1]

    nan_mask = np.isnan(residuals_full)
    if nan_mask.any():
        first_valid = np.where(~nan_mask)[0][0]
        residuals_full[:first_valid] = residuals_full[first_valid]

    return residuals_full


def find_label_transitions(labels):
    """Find label transition points, return OrderedDict {start_index: label_value}."""
    segments = OrderedDict()
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            segments[start] = labels[start]
            start = i
    segments[start] = labels[start]
    return segments


def compute_metrics(stat, labels_aligned, h, ic_len):
    """
    Compute detection performance metrics: FPR, FNR, F1, Detection Delay.
    """
    alarm = ((stat > h) | (stat < -h)).astype(int)
    truth = (labels_aligned > 1).astype(int)

    tp = np.sum((alarm == 1) & (truth == 1))
    fp = np.sum((alarm == 1) & (truth == 0))
    fn = np.sum((alarm == 0) & (truth == 1))
    tn = np.sum((alarm == 0) & (truth == 0))

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    oc_start = ic_len
    delay = -1
    if oc_start < len(alarm):
        for i in range(oc_start, len(alarm)):
            if alarm[i] == 1:
                delay = i - oc_start
                break

    return {"fpr": fpr, "fnr": fnr, "f1": f1, "delay": delay,
            "precision": prec, "recall": rec}


def draw_background(ax, time_axis, labels_aligned, label_colors):
    """Draw label region background colors and transition lines in subplot."""
    n = len(labels_aligned)
    segments = find_label_transitions(labels_aligned)
    for start_idx, lbl in segments.items():
        end_idx = min(start_idx + 1, n)
        for j in range(start_idx + 1, n):
            if labels_aligned[j] != lbl:
                end_idx = j
                break
        else:
            end_idx = n
        color = label_colors.get(lbl, "#eeeeee")
        ax.axvspan(time_axis[start_idx], time_axis[end_idx - 1],
                   color=color, alpha=0.35, linewidth=0)
    for start_idx, lbl in segments.items():
        if lbl != 1:
            ax.axvline(x=time_axis[start_idx], color="black",
                       linestyle=":", linewidth=0.7, alpha=0.5)


def get_arima_ewma_stats(y_all_orig, labels, window_size, ic_win_len, lam=EWMA_LAMBDA):
    """
    Compute ARIMA-EWMA statistics for the full dataset.
    Fit ARIMA on IC data, compute residuals for all data, then apply EWMA.
    """
    ic_count = np.sum(labels == 1)
    y_full = y_all_orig.flatten()
    arima_resid = forecast_arima_resid(
        None, y_full, ic_count, order=ARIMA_ORDER)

    arima_resid_aligned = arima_resid[window_size:]
    arima_resid_ic = arima_resid_aligned[:ic_win_len]

    mu_ar = np.mean(arima_resid_ic)
    sigma_ar = np.std(arima_resid_ic, ddof=1)
    e_std_ar = (arima_resid_aligned - mu_ar) / sigma_ar
    e_std_ar_ic = e_std_ar[:ic_win_len]

    Z_ar = compute_ewma(e_std_ar, lam=lam)
    Z_ar_ic = Z_ar[:ic_win_len]
    h_ar = 3 * np.std(Z_ar_ic, ddof=1)

    return Z_ar, h_ar, e_std_ar, e_std_ar_ic


# ============================================================
# Experiment 1: Baseline Comparison
# ============================================================
def experiment_baseline(prep, X_all_s, y_all_s, labels, model, ic_win_len):
    """
    Compare three methods:
      1. ARIMA-EWMA: ARIMA residuals + EWMA (classical statistical baseline)
      2. LSTM + Fixed Threshold: LSTM residuals, no EWMA smoothing
      3. LSTM-EWMA: proposed method
    """
    print("\n" + "=" * 70)
    print("  Exp1: Baseline Comparison")
    print("=" * 70)

    window_size = WINDOW_SIZE
    labels_aligned = labels[window_size:]

    y_pred_scaled = predict(model, create_dataset(
        X_all_s, y_all_s, window_size)[0])
    y_true_scaled = create_dataset(X_all_s, y_all_s, window_size)[1]

    y_pred_orig = prep.scaler_y.inverse_transform(y_pred_scaled)
    y_true_orig = prep.scaler_y.inverse_transform(y_true_scaled)
    residual = (y_true_orig - y_pred_orig).flatten()

    residual_ic = residual[:ic_win_len]
    mu_ic = np.mean(residual_ic)
    sigma_ic = np.std(residual_ic, ddof=1)
    e_std = (residual - mu_ic) / sigma_ic
    e_std_ic = e_std[:ic_win_len]

    # --- Method 1: ARIMA-EWMA ---
    print("  Fitting ARIMA model on IC data ...")
    y_orig_all = prep.scaler_y.inverse_transform(y_all_s)
    Z_arima, h_arima, _, _ = get_arima_ewma_stats(
        y_orig_all, labels, window_size, ic_win_len, lam=EWMA_LAMBDA)
    metrics_arima = compute_metrics(
        Z_arima, labels_aligned, h_arima, ic_win_len)

    # --- Method 2: LSTM + Fixed Threshold ---
    h_lstm = 3 * np.std(e_std_ic, ddof=1)
    metrics_lstm = compute_metrics(e_std, labels_aligned, h_lstm, ic_win_len)

    # --- Method 3: LSTM-EWMA (proposed) ---
    Z_ewma = compute_ewma(e_std, lam=EWMA_LAMBDA)
    Z_ic = Z_ewma[:ic_win_len]
    h_ewma = 3 * np.std(Z_ic, ddof=1)
    metrics_ewma = compute_metrics(
        Z_ewma, labels_aligned, h_ewma, ic_win_len)

    print(f"\n  ARIMA-EWMA:       FPR={metrics_arima['fpr']:.4f}  FNR={metrics_arima['fnr']:.4f}  "
          f"F1={metrics_arima['f1']:.4f}  Delay={metrics_arima['delay']}")
    print(f"  LSTM+Threshold:   FPR={metrics_lstm['fpr']:.4f}  FNR={metrics_lstm['fnr']:.4f}  "
          f"F1={metrics_lstm['f1']:.4f}  Delay={metrics_lstm['delay']}")
    print(f"  LSTM-EWMA:        FPR={metrics_ewma['fpr']:.4f}  FNR={metrics_ewma['fnr']:.4f}  "
          f"F1={metrics_ewma['f1']:.4f}  Delay={metrics_ewma['delay']}")

    # --- Comparison control chart (3 subplots) ---
    label_colors = {1: "#d4edda", 2: "#fff3cd",
                    3: "#ffd6cc", 4: "#f5c6cb", 5: "#ed969e"}
    time_axis = np.arange(window_size, window_size + len(labels_aligned))

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    datasets = [
        (Z_arima, h_arima, "ARIMA-EWMA (Classical Statistical)", metrics_arima),
        (e_std, h_lstm, "LSTM + Fixed Threshold (No EWMA)", metrics_lstm),
        (Z_ewma, h_ewma, "LSTM-EWMA (Proposed Method)", metrics_ewma),
    ]

    for ax, (stat, h_val, title, m) in zip(axes, datasets):
        draw_background(ax, time_axis, labels_aligned, label_colors)
        ax.plot(time_axis, stat, color="#2c3e50", linewidth=0.6)
        ax.axhline(y=h_val, color="red", linestyle="--", linewidth=1.0)
        ax.axhline(y=-h_val, color="red", linestyle="--", linewidth=1.0)
        ax.axhline(y=0, color="grey", linestyle=":", linewidth=0.4, alpha=0.5)

        ooc = (stat > h_val) | (stat < -h_val)
        if ooc.any():
            ax.scatter(time_axis[ooc], stat[ooc],
                       color="red", s=8, zorder=5)

        ax.set_ylabel("Statistic", fontsize=10)
        ax.set_title(f"{title}  |  FPR={m['fpr']:.3f}  FNR={m['fnr']:.3f}  "
                     f"F1={m['f1']:.3f}  Delay={m['delay']}", fontsize=11)

    axes[-1].set_xlabel("Time Step", fontsize=12)
    fig.suptitle(
        "Baseline Comparison: Control Charts of Three Methods", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig("exp1_baseline_comparison.png", dpi=200, bbox_inches="tight")
    print("  [Fig] Baseline comparison chart saved: exp1_baseline_comparison.png")
    plt.show()

    # --- Performance comparison table ---
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.axis("off")
    col_labels = ["Method", "FPR", "FNR", "F1-Score", "Detection Delay"]
    row_data = [
        ["ARIMA-EWMA", f"{metrics_arima['fpr']:.4f}", f"{metrics_arima['fnr']:.4f}",
         f"{metrics_arima['f1']:.4f}", f"{metrics_arima['delay']}"],
        ["LSTM + Threshold", f"{metrics_lstm['fpr']:.4f}", f"{metrics_lstm['fnr']:.4f}",
         f"{metrics_lstm['f1']:.4f}", f"{metrics_lstm['delay']}"],
        ["LSTM-EWMA (Proposed)", f"{metrics_ewma['fpr']:.4f}", f"{metrics_ewma['fnr']:.4f}",
         f"{metrics_ewma['f1']:.4f}", f"{metrics_ewma['delay']}"],
    ]
    table = ax2.table(cellText=row_data, colLabels=col_labels,
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif key[0] == 3:
            cell.set_facecolor("#E2EFDA")
    ax2.set_title("Detection Performance Comparison", fontsize=14, pad=20)
    fig2.savefig("exp1_performance_table.png", dpi=200, bbox_inches="tight")
    print("  [Table] Performance table saved: exp1_performance_table.png")
    plt.show()

    return {
        "e_std": e_std, "e_std_ic": e_std_ic,
        "Z_ewma": Z_ewma, "Z_ic": Z_ic, "h_ewma": h_ewma,
        "labels_aligned": labels_aligned, "ic_win_len": ic_win_len,
        "residual": residual,
    }


# ============================================================
# Experiment 2: Sensitivity Analysis
# ============================================================
def experiment_sensitivity(prep, X_all_s, y_all_s, labels, model, ic_win_len,
                           base_e_std, base_e_std_ic):
    """
    Sensitivity analysis:
      - Vary lambda and observe FPR / F1 / Delay
      - Vary window_size (requires retraining LSTM)
      - ROC curves for different methods
    """
    print("\n" + "=" * 70)
    print("  Exp2: Sensitivity Analysis")
    print("=" * 70)

    window_size = WINDOW_SIZE
    labels_aligned = labels[window_size:]

    # --- 2a: Lambda sensitivity ---
    lambda_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
    lambda_results = []

    print("\n  [2a] Lambda sensitivity:")
    for lam in lambda_values:
        Z = compute_ewma(base_e_std, lam=lam)
        Z_ic = Z[:ic_win_len]
        h = 3 * np.std(Z_ic, ddof=1)
        m = compute_metrics(Z, labels_aligned, h, ic_win_len)
        lambda_results.append({"lambda": lam, **m})
        print(f"    lambda={lam:.2f}  FPR={m['fpr']:.4f}  FNR={m['fnr']:.4f}  "
              f"F1={m['f1']:.4f}  Delay={m['delay']}")

    # --- 2b: Window Size sensitivity ---
    window_values = [5, 8, 10, 14, 20]
    window_results = []
    input_size = X_all_s.shape[1]

    print("\n  [2b] Window Size sensitivity (retraining required):")
    for ws in window_values:
        set_seed(SEED)
        ic_mask = labels == 1
        X_ic_s = X_all_s[ic_mask]
        y_ic_s = y_all_s[ic_mask]
        X_tr_win, y_tr_win = create_dataset(X_ic_s, y_ic_s, ws)
        X_te_win, y_te_win = create_dataset(X_all_s, y_all_s, ws)

        ws_ic_len = X_tr_win.shape[0]
        labels_ws = labels[ws:]

        m_lstm = train_lstm(X_tr_win, y_tr_win, input_size=input_size,
                            hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYERS,
                            dropout=LSTM_DROPOUT, epochs=EPOCHS,
                            batch_size=BATCH_SIZE, lr=LR, patience=PATIENCE)
        y_pred_s = predict(m_lstm, X_te_win)
        y_true_s = y_te_win
        y_pred_o = prep.scaler_y.inverse_transform(y_pred_s)
        y_true_o = prep.scaler_y.inverse_transform(y_true_s)
        res = (y_true_o - y_pred_o).flatten()
        res_ic = res[:ws_ic_len]
        mu_r = np.mean(res_ic)
        sig_r = np.std(res_ic, ddof=1)
        e_s = (res - mu_r) / sig_r
        e_s_ic = e_s[:ws_ic_len]

        Z_ws = compute_ewma(e_s, lam=EWMA_LAMBDA)
        Z_ws_ic = Z_ws[:ws_ic_len]
        h_ws = 3 * np.std(Z_ws_ic, ddof=1)
        m_ws = compute_metrics(Z_ws, labels_ws, h_ws, ws_ic_len)
        window_results.append({"window": ws, "ic_len": ws_ic_len, **m_ws})
        print(f"    window={ws:>2d}  FPR={m_ws['fpr']:.4f}  FNR={m_ws['fnr']:.4f}  "
              f"F1={m_ws['f1']:.4f}  Delay={m_ws['delay']}")

    # --- 2c: ROC curves ---
    print("\n  [2c] ROC curve computation ...")
    roc_data = {}

    Z_base = compute_ewma(base_e_std, lam=EWMA_LAMBDA)
    truth = (labels_aligned > 1).astype(int)
    abs_Z = np.abs(Z_base)
    fpr_t, tpr_t, _ = roc_curve(truth, abs_Z)
    auc_t = auc(fpr_t, tpr_t)
    roc_data["LSTM-EWMA"] = (fpr_t, tpr_t, auc_t)

    abs_e = np.abs(base_e_std)
    fpr_e, tpr_e, _ = roc_curve(truth, abs_e)
    auc_e = auc(fpr_e, tpr_e)
    roc_data["LSTM + Threshold"] = (fpr_e, tpr_e, auc_e)

    y_orig_all = prep.scaler_y.inverse_transform(y_all_s)
    Z_arima, _, e_std_ar, _ = get_arima_ewma_stats(
        y_orig_all, labels, window_size, ic_win_len, lam=EWMA_LAMBDA)
    abs_za = np.abs(Z_arima)
    fpr_a, tpr_a, _ = roc_curve(truth, abs_za)
    auc_a = auc(fpr_a, tpr_a)
    roc_data["ARIMA-EWMA"] = (fpr_a, tpr_a, auc_a)

    for name, (_, _, a) in roc_data.items():
        print(f"    {name}: AUC = {a:.4f}")

    # --- Plot: Lambda sensitivity ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    lam_arr = [r["lambda"] for r in lambda_results]
    axes[0].plot(lam_arr, [r["fpr"] for r in lambda_results], "o-",
                 color="#e74c3c", linewidth=2, markersize=7)
    axes[0].set_xlabel("lambda", fontsize=12)
    axes[0].set_ylabel("FPR (False Positive Rate)", fontsize=11)
    axes[0].set_title("FPR vs lambda", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(lam_arr, [r["f1"] for r in lambda_results], "s-",
                 color="#2ecc71", linewidth=2, markersize=7)
    axes[1].set_xlabel("lambda", fontsize=12)
    axes[1].set_ylabel("F1-Score", fontsize=11)
    axes[1].set_title("F1-Score vs lambda", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    delay_arr = [r["delay"] for r in lambda_results]
    axes[2].plot(lam_arr, delay_arr, "D-",
                 color="#3498db", linewidth=2, markersize=7)
    axes[2].set_xlabel("lambda", fontsize=12)
    axes[2].set_ylabel("Detection Delay (steps)", fontsize=11)
    axes[2].set_title("Detection Delay vs lambda", fontsize=12)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Sensitivity Analysis: Effect of lambda", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig("exp2_lambda_sensitivity.png", dpi=200, bbox_inches="tight")
    print("  [Fig] Lambda sensitivity saved: exp2_lambda_sensitivity.png")
    plt.show()

    # --- Plot: Window Size sensitivity ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5))

    ws_arr = [r["window"] for r in window_results]
    axes2[0].plot(ws_arr, [r["fpr"] for r in window_results], "o-",
                  color="#e74c3c", linewidth=2, markersize=7)
    axes2[0].set_xlabel("Window Size", fontsize=12)
    axes2[0].set_ylabel("FPR (False Positive Rate)", fontsize=11)
    axes2[0].set_title("FPR vs Window Size", fontsize=12)
    axes2[0].grid(True, alpha=0.3)

    axes2[1].plot(ws_arr, [r["f1"] for r in window_results], "s-",
                  color="#2ecc71", linewidth=2, markersize=7)
    axes2[1].set_xlabel("Window Size", fontsize=12)
    axes2[1].set_ylabel("F1-Score", fontsize=11)
    axes2[1].set_title("F1-Score vs Window Size", fontsize=12)
    axes2[1].grid(True, alpha=0.3)

    axes2[2].plot(ws_arr, [r["delay"] for r in window_results], "D-",
                  color="#3498db", linewidth=2, markersize=7)
    axes2[2].set_xlabel("Window Size", fontsize=12)
    axes2[2].set_ylabel("Detection Delay (steps)", fontsize=11)
    axes2[2].set_title("Detection Delay vs Window Size", fontsize=12)
    axes2[2].grid(True, alpha=0.3)

    fig2.suptitle("Sensitivity Analysis: Effect of Window Size",
                  fontsize=14, y=1.02)
    fig2.tight_layout()
    fig2.savefig("exp2_window_sensitivity.png", dpi=200, bbox_inches="tight")
    print("  [Fig] Window sensitivity saved: exp2_window_sensitivity.png")
    plt.show()

    # --- Plot: ROC curves ---
    fig3, ax3 = plt.subplots(figsize=(7, 7))
    colors_roc = {"LSTM-EWMA": "#e74c3c", "LSTM + Threshold": "#3498db",
                  "ARIMA-EWMA": "#95a5a6"}
    for name, (fpr_v, tpr_v, auc_v) in roc_data.items():
        ax3.plot(fpr_v, tpr_v, linewidth=2.2, color=colors_roc.get(name, "#333"),
                 label=f"{name} (AUC={auc_v:.4f})")
    ax3.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax3.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax3.set_ylabel("True Positive Rate (TPR)", fontsize=12)
    ax3.set_title("ROC Curve Comparison", fontsize=14)
    ax3.legend(fontsize=11, loc="lower right")
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig("exp2_roc_curve.png", dpi=200, bbox_inches="tight")
    print("  [Fig] ROC curve saved: exp2_roc_curve.png")
    plt.show()

    return lambda_results, window_results


# ============================================================
# Experiment 3: Ablation Study
# ============================================================
def experiment_ablation(prep, X_all_s, y_all_s, labels, model, ic_win_len,
                        base_e_std, base_e_std_ic):
    """
    Ablation study:
      1. ARIMA-EWMA: no LSTM, ARIMA residuals + EWMA
      2. LSTM-only: LSTM residuals, no EWMA smoothing
      3. LSTM-EWMA: full model
    """
    print("\n" + "=" * 70)
    print("  Exp3: Ablation Study")
    print("=" * 70)

    window_size = WINDOW_SIZE
    labels_aligned = labels[window_size:]

    # --- Ablation 1: ARIMA-EWMA (no LSTM) ---
    print("  Fitting ARIMA model for ablation ...")
    y_orig_all = prep.scaler_y.inverse_transform(y_all_s)
    Z_arima, h_arima, _, _ = get_arima_ewma_stats(
        y_orig_all, labels, window_size, ic_win_len, lam=EWMA_LAMBDA)
    m_arima = compute_metrics(Z_arima, labels_aligned, h_arima, ic_win_len)

    # --- Ablation 2: LSTM-only (residuals, no EWMA) ---
    h_lstm_only = 3 * np.std(base_e_std_ic, ddof=1)
    m_lstm_only = compute_metrics(
        base_e_std, labels_aligned, h_lstm_only, ic_win_len)

    # --- Ablation 3: LSTM-EWMA (full model) ---
    Z_full = compute_ewma(base_e_std, lam=EWMA_LAMBDA)
    Z_full_ic = Z_full[:ic_win_len]
    h_full = 3 * np.std(Z_full_ic, ddof=1)
    m_full = compute_metrics(Z_full, labels_aligned, h_full, ic_win_len)

    print(f"\n  {'Method':<20s} {'LSTM':<8s} {'EWMA':<8s} {'FPR':<10s} {'FNR':<10s} "
          f"{'F1':<10s} {'Delay':<8s}")
    print("  " + "-" * 70)
    print(f"  {'ARIMA-EWMA':<20s} {'x':<8s} {'v':<8s} {m_arima['fpr']:<10.4f} "
          f"{m_arima['fnr']:<10.4f} {m_arima['f1']:<10.4f} {m_arima['delay']:<8d}")
    print(f"  {'LSTM-only':<20s} {'v':<8s} {'x':<8s} {m_lstm_only['fpr']:<10.4f} "
          f"{m_lstm_only['fnr']:<10.4f} {m_lstm_only['f1']:<10.4f} {m_lstm_only['delay']:<8d}")
    print(f"  {'LSTM-EWMA':<20s} {'v':<8s} {'v':<8s} {m_full['fpr']:<10.4f} "
          f"{m_full['fnr']:<10.4f} {m_full['f1']:<10.4f} {m_full['delay']:<8d}")

    # --- Ablation comparison control chart ---
    label_colors = {1: "#d4edda", 2: "#fff3cd",
                    3: "#ffd6cc", 4: "#f5c6cb", 5: "#ed969e"}
    time_axis = np.arange(window_size, window_size + len(labels_aligned))

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    ablation_data = [
        (Z_arima, h_arima, "ARIMA-EWMA (No LSTM)", m_arima),
        (base_e_std, h_lstm_only, "LSTM-only (No EWMA Smoothing)", m_lstm_only),
        (Z_full, h_full, "LSTM-EWMA (Full Model)", m_full),
    ]

    for ax, (stat, h_val, title, m) in zip(axes, ablation_data):
        draw_background(ax, time_axis, labels_aligned, label_colors)
        ax.plot(time_axis, stat, color="#2c3e50", linewidth=0.6)
        ax.axhline(y=h_val, color="red", linestyle="--", linewidth=1.0)
        ax.axhline(y=-h_val, color="red", linestyle="--", linewidth=1.0)
        ax.axhline(y=0, color="grey", linestyle=":", linewidth=0.4, alpha=0.5)
        ooc = (stat > h_val) | (stat < -h_val)
        if ooc.any():
            ax.scatter(time_axis[ooc], stat[ooc], color="red", s=8, zorder=5)
        ax.set_ylabel("Statistic", fontsize=10)
        ax.set_title(f"{title}  |  FPR={m['fpr']:.3f}  FNR={m['fnr']:.3f}  "
                     f"F1={m['f1']:.3f}  Delay={m['delay']}", fontsize=11)

    axes[-1].set_xlabel("Time Step", fontsize=12)
    fig.suptitle(
        "Ablation Study: Contribution of Each Module", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig("exp3_ablation_comparison.png", dpi=200, bbox_inches="tight")
    print("  [Fig] Ablation comparison saved: exp3_ablation_comparison.png")
    plt.show()

    # --- Ablation performance table ---
    fig2, ax2 = plt.subplots(figsize=(11, 3.5))
    ax2.axis("off")
    col_labels = ["Method", "LSTM", "EWMA", "FPR", "FNR", "F1-Score", "Delay"]
    row_data = [
        ["ARIMA-EWMA", "x", "v",
         f"{m_arima['fpr']:.4f}", f"{m_arima['fnr']:.4f}",
         f"{m_arima['f1']:.4f}", f"{m_arima['delay']}"],
        ["LSTM-only", "v", "x",
         f"{m_lstm_only['fpr']:.4f}", f"{m_lstm_only['fnr']:.4f}",
         f"{m_lstm_only['f1']:.4f}", f"{m_lstm_only['delay']}"],
        ["LSTM-EWMA", "v", "v",
         f"{m_full['fpr']:.4f}", f"{m_full['fnr']:.4f}",
         f"{m_full['f1']:.4f}", f"{m_full['delay']}"],
    ]
    table = ax2.table(cellText=row_data, colLabels=col_labels,
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif key[0] == 3:
            cell.set_facecolor("#E2EFDA")
        if key[0] > 0 and key[1] in [1, 2]:
            val = row_data[key[0] - 1][key[1]]
            if val == "v":
                cell.set_facecolor("#C6EFCE")
            elif val == "x":
                cell.set_facecolor("#FFC7CE")
    ax2.set_title("Ablation Study: Performance Comparison",
                  fontsize=14, pad=20)
    fig2.savefig("exp3_ablation_table.png", dpi=200, bbox_inches="tight")
    print("  [Table] Ablation table saved: exp3_ablation_table.png")
    plt.show()

    return m_arima, m_lstm_only, m_full


# ============================================================
# Main
# ============================================================
def main():
    set_seed(SEED)
    print("=" * 70)
    print("  LSTM-EWMA Residual Anomaly Detection - Extended Experiments")
    print("=" * 70)

    print("\n[Data Preparation] Loading and preprocessing ...")
    prep = DataPreparer(DATA_PATH, target_col=TARGET_COL)
    X_all, y_all, labels = prep.load()
    (X_ic, y_ic), (X_all_s, y_all_s) = prep.split_and_scale(
        X_all, y_all, labels)
    print(f"  Total data: {X_all.shape[0]} rows, {X_all.shape[1]} features")
    print(f"  IC data:    {X_ic.shape[0]} rows")

    print(f"\n[LSTM Training] window_size={WINDOW_SIZE} ...")
    X_train_win, y_train_win = create_dataset(X_ic, y_ic, WINDOW_SIZE)
    X_test_win, y_test_win = create_dataset(X_all_s, y_all_s, WINDOW_SIZE)
    ic_win_len = X_train_win.shape[0]
    input_size = X_train_win.shape[2]

    model = train_lstm(
        X_train_win, y_train_win, input_size=input_size,
        hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYERS,
        dropout=LSTM_DROPOUT, epochs=EPOCHS,
        batch_size=BATCH_SIZE, lr=LR, patience=PATIENCE,
    )

    y_pred_scaled = predict(model, X_test_win)
    y_pred_orig = prep.scaler_y.inverse_transform(y_pred_scaled)
    y_true_orig = prep.scaler_y.inverse_transform(y_test_win)
    residual = (y_true_orig - y_pred_orig).flatten()
    residual_ic = residual[:ic_win_len]
    mu_ic = np.mean(residual_ic)
    sigma_ic = np.std(residual_ic, ddof=1)
    e_std = (residual - mu_ic) / sigma_ic
    e_std_ic = e_std[:ic_win_len]

    print(f"  IC residual mean: {mu_ic:.4f}, std: {sigma_ic:.4f}")

    # ---- Exp1: Baseline Comparison ----
    experiment_baseline(prep, X_all_s, y_all_s, labels, model, ic_win_len)

    # ---- Exp2: Sensitivity Analysis ----
    experiment_sensitivity(prep, X_all_s, y_all_s, labels, model, ic_win_len,
                           e_std, e_std_ic)

    # ---- Exp3: Ablation Study ----
    experiment_ablation(prep, X_all_s, y_all_s, labels, model, ic_win_len,
                        e_std, e_std_ic)

    print("\n" + "=" * 70)
    print("  [Done] All experiments completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
