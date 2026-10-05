import numpy as np
import pandas as pd
from pathlib import Path
# ---------------------------------------------------------------------------
# Preparing the dataset
# ---------------------------------------------------------------------------
COLUMNS = [
    "engine_displacement",
    "horsepower",
    "vehicle_weight",
    "model_year",
    "fuel_efficiency_mpg",
]

csv_path = Path(__file__).parent / 'car_fuel_efficiency_2026.csv'
df = pd.read_csv(csv_path)
df = df[COLUMNS]

TARGET = "fuel_efficiency_mpg"


def split_data(df, seed):
    """Shuffle + 60/20/20 train/val/test split exactly as in the lecture."""
    n = len(df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test

    np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:n_train]].reset_index(drop=True)
    df_val = df.iloc[idx[n_train:n_train + n_val]].reset_index(drop=True)
    df_test = df.iloc[idx[n_train + n_val:]].reset_index(drop=True)

    return df_train, df_val, df_test


def prepare_X(df, fill_value):
    """Feature matrix with NAs filled by `fill_value`."""
    df = df.copy()
    df = df.fillna(fill_value)
    return df[[c for c in COLUMNS if c != TARGET]].values


def train_linear_regression(X, y, r=0.0):
    """Normal equation with optional L2 regularization `r`."""
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]


def predict(X, w0, w):
    return w0 + X.dot(w)


def rmse(y, y_pred):
    return np.sqrt(np.mean((y - y_pred) ** 2))


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
print("=" * 60)
print("EDA: distribution of fuel_efficiency_mpg")
print("=" * 60)
print(df[TARGET].describe())
# A long tail would show mean clearly above the median / large skew.
print(f"Skew: {df[TARGET].skew():.4f}")
print("-> Roughly symmetric (skew ~ 0): it does NOT have a long tail.\n")


# ---------------------------------------------------------------------------
# Question 1: which column has missing values?
# ---------------------------------------------------------------------------
print("=" * 60)
print("Question 1: column with missing values")
print("=" * 60)
na_counts = df.isnull().sum()
print(na_counts)
q1 = na_counts[na_counts > 0].index.tolist()
print(f"ANSWER Q1: {q1}\n")


# ---------------------------------------------------------------------------
# Question 2: median of horsepower
# ---------------------------------------------------------------------------
print("=" * 60)
print("Question 2: median of 'horsepower'")
print("=" * 60)
q2 = df["horsepower"].median()
print(f"ANSWER Q2: {q2}\n")


# ---------------------------------------------------------------------------
# Question 3: fill NA with 0 vs mean (seed 42, no regularization)
# ---------------------------------------------------------------------------
print("=" * 60)
print("Question 3: fill NA with 0 vs mean")
print("=" * 60)
df_train, df_val, df_test = split_data(df, seed=42)

y_train = df_train[TARGET].values
y_val = df_val[TARGET].values

# Option A: fill with 0
X_train_0 = prepare_X(df_train, 0)
X_val_0 = prepare_X(df_val, 0)
w0_0, w_0 = train_linear_regression(X_train_0, y_train)
rmse_0 = round(rmse(y_val, predict(X_val_0, w0_0, w_0)), 3)

# Option B: fill with mean (computed on TRAIN only)
hp_mean = df_train["horsepower"].mean()
X_train_mean = prepare_X(df_train, hp_mean)
X_val_mean = prepare_X(df_val, hp_mean)
w0_m, w_m = train_linear_regression(X_train_mean, y_train)
rmse_mean = round(rmse(y_val, predict(X_val_mean, w0_m, w_m)), 3)

print(f"RMSE with 0:    {rmse_0}")
print(f"RMSE with mean: {rmse_mean}")
if rmse_0 < rmse_mean:
    q3 = "With 0"
elif rmse_mean < rmse_0:
    q3 = "With mean"
else:
    q3 = "Both are equally good"
print(f"ANSWER Q3: {q3}\n")


# ---------------------------------------------------------------------------
# Question 4: regularized regression, fill NA with 0, try several r
# ---------------------------------------------------------------------------
print("=" * 60)
print("Question 4: best r for regularized regression")
print("=" * 60)
r_values = [0, 0.01, 0.1, 1, 5, 10, 100]
results_q4 = []
for r in r_values:
    w0_r, w_r = train_linear_regression(X_train_0, y_train, r=r)
    score = round(rmse(y_val, predict(X_val_0, w0_r, w_r)), 4)
    results_q4.append((r, score))
    print(f"r={r:<6} RMSE={score}")

best_score = min(s for _, s in results_q4)
q4 = min(r for r, s in results_q4 if s == best_score)  # smallest r on tie
print(f"ANSWER Q4: r = {q4}\n")


# ---------------------------------------------------------------------------
# Question 5: std of RMSE across seeds 0..9 (fill 0, no regularization)
# ---------------------------------------------------------------------------
print("=" * 60)
print("Question 5: std of validation RMSE across seeds")
print("=" * 60)
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
scores_q5 = []
for seed in seeds:
    d_tr, d_val, d_te = split_data(df, seed=seed)
    y_tr = d_tr[TARGET].values
    y_v = d_val[TARGET].values
    X_tr = prepare_X(d_tr, 0)
    X_v = prepare_X(d_val, 0)
    w0_s, w_s = train_linear_regression(X_tr, y_tr)
    score = rmse(y_v, predict(X_v, w0_s, w_s))
    scores_q5.append(score)
    print(f"seed={seed} RMSE={score:.6f}")

q5 = round(np.std(scores_q5), 3)
print(f"ANSWER Q5: std = {q5}\n")


# ---------------------------------------------------------------------------
# Question 6: seed 9, combine train+val, fill 0, r=0.001, RMSE on test
# ---------------------------------------------------------------------------
print("=" * 60)
print("Question 6: RMSE on test set")
print("=" * 60)
d_tr, d_val, d_te = split_data(df, seed=9)

df_full_train = pd.concat([d_tr, d_val]).reset_index(drop=True)
y_full_train = df_full_train[TARGET].values
y_test = d_te[TARGET].values

X_full_train = prepare_X(df_full_train, 0)
X_test = prepare_X(d_te, 0)

w0_6, w_6 = train_linear_regression(X_full_train, y_full_train, r=0.001)
q6 = round(rmse(y_test, predict(X_test, w0_6, w_6)), 3)
print(f"ANSWER Q6: RMSE on test = {q6}\n")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("=" * 60)
print("SUMMARY OF ANSWERS")
print("=" * 60)
print(f"Q1: {q1}")
print(f"Q2: {q2}")
print(f"Q3: {q3}")
print(f"Q4: {q4}")
print(f"Q5: {q5}")
print(f"Q6: {q6}")
