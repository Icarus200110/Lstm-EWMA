"""
EWMA control chart statistics and control limit determination.
"""

import numpy as np


def compute_ewma(residuals, lam=0.05):
    """
    Compute EWMA statistic sequence.
    Z_t = lambda * e_t + (1 - lambda) * Z_{t-1}
    """
    Z = np.zeros_like(residuals, dtype=np.float64)
    Z[0] = residuals[0]
    for t in range(1, len(residuals)):
        Z[t] = (1 - lam) * Z[t - 1] + lam * residuals[t]
    return Z


def compute_control_limit(e_std_ic, lam=0.05, k=3):
    """
    Compute EWMA control limit width h based on IC phase residuals.
    Uses 3-sigma rule on the IC EWMA statistics.
    """
    Z_ic = compute_ewma(e_std_ic, lam)
    h = k * np.std(Z_ic, ddof=1)
    return h


def determine_control_limit_bilateral(ic_residuals, lam=0.05, target_arl0=400, n_sim=5000, seq_len=500):
    """
    Determine bilateral control limit h via Monte Carlo simulation.
    Returns h such that the in-control ARL matches target_arl0.
    """
    print(
        f"  Determining control limit (lambda={lam}, target ARL0={target_arl0})...")

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

    # Binary search for h
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
