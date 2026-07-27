"""
Experiment utilities: metrics computation, ARIMA baseline, and visualization helpers.
"""

import numpy as np
from collections import OrderedDict


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
    from statsmodels.tsa.arima.model import ARIMA

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


def get_arima_ewma_stats(y_all_orig, labels, window_size, ic_win_len, lam=0.05, arima_order=(2, 1, 2)):
    """
    Compute ARIMA-EWMA statistics for the full dataset.
    Fit ARIMA on IC data, compute residuals for all data, then apply EWMA.
    """
    ic_count = np.sum(labels == 1)
    y_full = y_all_orig.flatten()
    arima_resid = forecast_arima_resid(
        None, y_full, ic_count, order=arima_order)

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
