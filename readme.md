# LSTM-EWMA Anomaly Detection for DNA sequencing Data

## Project Structure

```
src/
├── training/          # Core training pipeline (v3 Poisson loss)
│   ├── __init__.py
│   ├── model.py       # LSTM-Poisson model
│   ├── data.py        # Data loading and preprocessing
│   ├── residuals.py   # Residual computation (Pearson, Deviance, etc.)
│   ├── ewma.py        # EWMA statistics and control limits
│   └── plots.py       # Academic-style plotting functions
├── experiments/       # Extended experiments (MSE loss + ARIMA baseline)
│   ├── __init__.py
│   ├── arima.py       # ARIMA-EWMA baseline
│   ├── metrics.py     # Detection metrics (FPR, FNR, F1, Delay, AUC)
│   ├── baseline.py    # Exp1: ARIMA-EWMA vs LSTM vs LSTM-EWMA
│   ├── sensitivity.py # Exp2: Lambda/window/ROC sensitivity
│   └── ablation.py    # Exp3: Module ablation study
└── utils.py           # Shared utilities
configs/
train.py               # Main training script (Poisson loss, runs full pipeline)
run_experiments.py     # Main experiment script (MSE loss, 3 experiments)
requirements.txt       # Python dependencies
```

## Quick Start

```bash
# 1. Training pipeline (v3, Poisson loss)
python train.py

# 2. Extended experiments (MSE loss + ARIMA baseline)
python run_experiments.py
```

## Data Format

`监控训练数据.csv` — no header row, 1510 rows, 9 columns:

- Col 0–5: covariates (float)
- Col 6: exposure volume (int)
- Col 7: response count (int, **no zero counts**)
- Col 8: label (1=IC, 2/3/4/5=OC types)

## Configuration

Key hyperparameters are defined at the top of each script:

| Parameter    | train.py             | run_experiments.py |
| ------------ | -------------------- | ------------------ |
| Window size  | 14                   | 10                 |
| Hidden units | 64                   | 64                 |
| LSTM layers  | 2                    | 2                  |
| Dropout      | 0.2                  | 0.2                |
| Loss         | Poisson NLL          | MSE                |
| EWMA lambda  | 0.01, 0.05, 0.1, 0.2 | 0.05               |
| ARIMA order  | —                    | (2,1,2)            |
| Epochs       | 200                  | 200                |

## Results

- `train.py` → `results_v4/` (8 figures + JSON + model checkpoint)
- `run_experiments.py` → current directory (exp1/exp2/exp3 figures)
