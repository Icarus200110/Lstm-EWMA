"""
Academic-style plotting functions for the training pipeline (QREI journal style).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# QREI academic style settings
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'SimSun']
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['figure.figsize'] = (8, 6)
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['lines.linewidth'] = 1.0
rcParams['axes.linewidth'] = 0.8
rcParams['grid.alpha'] = 0.3
rcParams['grid.linestyle'] = '--'
rcParams['grid.linewidth'] = 0.5


def plot_training_curve(train_losses, val_losses, output_path):
    """Plot training curve (QREI academic style)."""
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
    """Plot IC EWMA control chart (bilateral, QREI style)."""
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
    ax.set_title(
        f'(a) Phase I: IC EWMA Control Chart (lambda={lam})', fontsize=13)
    ax.legend(loc='upper right', frameon=True, edgecolor='black', ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_full_ewma(all_ewma, all_labels, h, lam, ic_mean, output_path):
    """Plot full data EWMA control chart (bilateral, QREI style)."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ic_mask = (all_labels == 1)
    oc_mask = (all_labels != 1)
    time_idx = np.arange(len(all_ewma))

    # Draw EWMA line (gray thin line)
    ax.plot(time_idx, all_ewma, 'gray', linewidth=0.4, alpha=0.5)

    # IC points (black small dots)
    ic_idx = np.where(ic_mask)[0]
    if len(ic_idx) > 0:
        ax.scatter(ic_idx, all_ewma[ic_mask], c='black',
                   s=3, alpha=0.3, label='IC', zorder=3)

    # OC points (red dots)
    oc_idx = np.where(oc_mask)[0]
    if len(oc_idx) > 0:
        ax.scatter(oc_idx, all_ewma[oc_mask], c='red',
                   s=5, alpha=0.6, label='OC', zorder=3)

    # Control limits and center line
    ax.axhline(y=ic_mean + h, color='r', linestyle='--',
               linewidth=1.2, label=f'UCL = {ic_mean+h:.3f}')
    ax.axhline(y=ic_mean - h, color='r', linestyle='--',
               linewidth=1.2, label=f'LCL = {ic_mean-h:.3f}')
    ax.axhline(y=ic_mean, color='k', linestyle=':',
               linewidth=0.8, alpha=0.7, label='Center line')

    # Alarm points (points beyond control limits)
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
    """Prediction analysis plot (2x2 subplots, QREI academic style)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ic_m = (all_labels == 1)
    oc_m = (all_labels != 1)

    # (a) Observed vs Predicted mean
    ax = axes[0, 0]
    ax.plot(all_y, 'k-', linewidth=0.6, alpha=0.5, label='Observed y')
    ax.plot(all_mu, 'r--', linewidth=0.6, alpha=0.5, label='Predicted mu')
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('(a) Observed vs Predicted Poisson Mean', fontsize=12)
    ax.legend(loc='upper right', frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3)

    # (b) Predicted log Poisson rate
    ax = axes[0, 1]
    ax.plot(all_log_lambda, 'k-', linewidth=0.6, alpha=0.7)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('log(lambda)', fontsize=11)
    ax.set_title('(b) Predicted Log Poisson Rate', fontsize=12)
    ax.grid(True, alpha=0.3)

    # (c) Residual sequence
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

    # (d) Predicted vs Observed scatter
    ax = axes[1, 1]
    ax.scatter(all_mu[ic_m], all_y[ic_m], c='black',
               alpha=0.2, s=6, label='IC')
    ax.scatter(all_mu[oc_m], all_y[oc_m], c='red',
               alpha=0.3, s=6, label='OC')
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
    """Residual distribution plot (QREI academic style, grayscale)."""
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
    """Per-type EWMA control chart (QREI academic style)."""
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
    """ARL and detection rate comparison chart (QREI academic style)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    oc_types = [2, 3, 4, 5]
    x = np.arange(len(oc_types))
    width = 0.2
    colors = ['#2c3e50', '#e74c3c', '#3498db', '#95a5a6']

    # (a) ARL comparison bar chart
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

    # (b) Detection rate comparison
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
    """Residual autocorrelation analysis chart (QREI style)."""
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
    # Confidence interval
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

    # Residual QQ plot
    ax = axes[1]
    from scipy import stats
    ic_res_sorted = np.sort(ic_res)
    n_ic = len(ic_res_sorted)
    theoretical = stats.norm.ppf(
        (np.arange(1, n_ic + 1) - 0.5) / n_ic)
    ax.scatter(theoretical, ic_res_sorted, c='black', s=8, alpha=0.5)
    # Reference line
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
