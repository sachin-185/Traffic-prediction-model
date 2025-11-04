import os
import math
import h5py
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# -----------------------------
# Quick device selection (GPU if available)
# -----------------------------
gpus = tf.config.list_physical_devices('GPU')
device = '/GPU:0' if gpus else '/CPU:0'
print(f"Using device: {device}")
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print("Enabled GPU memory growth.")
    except RuntimeError as e:
        print("Could not enable memory growth:", e)

# -----------------------------
# Config (kept small & sensible)
# -----------------------------
DATA_DIR = "data"
H5_FILENAME = "METR-LA.h5"
LOOKBACK = 12
HORIZON = 12
HORIZONS_TO_EVAL = [3, 6, 12]
BATCH_SIZE = 64
EPOCHS = 10
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
MODEL_DIR = "models_metr_la"
PRED_DIR = "predictions_metr_la"
OPTIMIZER_LR = 5e-4
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# Load METR-LA (robust)
# -----------------------------
def load_metr_la(path=os.path.join(DATA_DIR, H5_FILENAME)):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    data = None
    try:
        with h5py.File(path, "r") as f:
            for key in ['df/value', 'df', 'speed']:
                if key in f:
                    print(f"Access via h5py key: '{key}'")
                    data = f[key][()].transpose()
                    break
    except Exception as e:
        print("h5py load failed, trying pandas:", e)

    if data is None:
        try:
            print("Using pandas.read_hdf")
            df = pd.read_hdf(path)
            data = df.values
        except Exception as e:
            raise Exception(f"Failed to load HDF5: {e}")

    print("Loaded data shape:", data.shape)
    return data.astype(np.float32)

# -----------------------------
# Sequence helpers & baseline
# -----------------------------
def create_sequences_multivariate(data_scaled, lookback, horizon):
    X, Y = [], []
    n = len(data_scaled)
    for i in range(n - lookback - horizon + 1):
        X.append(data_scaled[i:i+lookback])
        Y.append(data_scaled[i+lookback:i+lookback+horizon])
    return np.array(X), np.array(Y)

def baseline_historical_avg(X_samples, horizon):
    last_mean = X_samples.mean(axis=1)
    return np.repeat(last_mean[:, np.newaxis, :], horizon, axis=1)

# -----------------------------
# Model builder (stacked LSTM)
# -----------------------------
def build_lstm_multivariate(lookback, n_sensors, horizon):
    model = Sequential([
        Input(shape=(lookback, n_sensors)),
        LSTM(256, return_sequences=True),
        Dropout(0.3),
        LSTM(128, return_sequences=False),
        Dropout(0.3),
        Dense(horizon * n_sensors),
        Reshape((horizon, n_sensors))
    ])
    opt = Adam(learning_rate=OPTIMIZER_LR)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model

# -----------------------------
# Scaling helpers & metrics
# -----------------------------
def inverse_scale_triplet(pred_scaled, scaler):
    s, h, ns = pred_scaled.shape
    flat = pred_scaled.reshape(-1, ns)
    inv_flat = scaler.inverse_transform(flat)
    return inv_flat.reshape(s, h, ns)

def compute_metrics_per_horizon(y_true, y_pred, horizons):
    results = {}
    eps = 1e-5
    for h in horizons:
        idx = h - 1
        true_h = y_true[:, idx, :].reshape(-1)
        pred_h = y_pred[:, idx, :].reshape(-1)
        mask = np.abs(true_h) > eps
        true_masked = true_h[mask]
        pred_masked = pred_h[mask]
        rmse = math.sqrt(mean_squared_error(true_h, pred_h))
        mae = mean_absolute_error(true_h, pred_h)
        mape = 0.0 if len(true_masked) == 0 else np.mean(np.abs((true_masked - pred_masked) / true_masked)) * 100
        results[h] = {"rmse": rmse, "mae": mae, "mape": mape}
    return results

def print_results_table(name, res_dict):
    print(f"\n--- {name} ---")
    print(f"{'Horizon':>7} | {'Min':>4} | {'RMSE':>8} | {'MAE':>8} | {'MAPE %':>8}")
    print("-"*50)
    for h in sorted(res_dict.keys()):
        mins = h * 5
        r = res_dict[h]
        print(f"{h:7d} | {mins:4d} | {r['rmse']:8.4f} | {r['mae']:8.4f} | {r['mape']:8.2f}")

# -----------------------------
# Main: LSTM-only pipeline (GPU if available)
# -----------------------------
def main():
    print("Loading data...")
    try:
        data = load_metr_la()
    except Exception as e:
        print("Fatal:", e)
        return

    data[np.isnan(data)] = 0.0
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, Y = create_sequences_multivariate(data_scaled, LOOKBACK, HORIZON)
    print("Sequence shapes X, Y:", X.shape, Y.shape)

    total = X.shape[0]
    train_end = int(total * TRAIN_RATIO)
    val_end = int(total * (TRAIN_RATIO + VAL_RATIO))
    X_train, Y_train = X[:train_end], Y[:train_end]
    X_val, Y_val = X[train_end:val_end], Y[train_end:val_end]
    X_test, Y_test = X[val_end:], Y[val_end:]

    print(f"samples -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    if len(X_test) == 0:
        print("Test split empty -> adjust ratios.")
        return

    n_sensors = X.shape[2]
    baseline_scaled = baseline_historical_avg(X_test, HORIZON)
    baseline = inverse_scale_triplet(baseline_scaled, scaler)

    # Build and train model on selected device
    model_path = os.path.join(MODEL_DIR, "lstm_best.keras")
    with tf.device(device):
        lstm = build_lstm_multivariate(LOOKBACK, n_sensors, HORIZON)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss")
    ]

    print(f"\nTraining LSTM for up to {EPOCHS} epochs on {device} ...")
    lstm.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2
    )

    # Inference & inverse scale
    print("Predicting on test set...")
    lstm_pred_scaled = lstm.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    lstm_pred = inverse_scale_triplet(lstm_pred_scaled, scaler)
    Y_test_true = inverse_scale_triplet(Y_test, scaler)

    # Save few predictions & truth
    N_SAVE = min(10, X_test.shape[0])
    for i in range(N_SAVE):
        pd.DataFrame(lstm_pred[i], index=[f"t+{k+1}" for k in range(HORIZON)]).to_csv(os.path.join(PRED_DIR, f"lstm_sample{i}.csv"))
        pd.DataFrame(Y_test_true[i], index=[f"t+{k+1}" for k in range(HORIZON)]).to_csv(os.path.join(PRED_DIR, f"true_sample{i}.csv"))

    # Evaluate
    baseline_res = compute_metrics_per_horizon(Y_test_true, baseline, HORIZONS_TO_EVAL)
    lstm_res = compute_metrics_per_horizon(Y_test_true, lstm_pred, HORIZONS_TO_EVAL)

    print_results_table("Baseline (Historical Avg)", baseline_res)
    print_results_table("STACKED LSTM", lstm_res)

    def mean_over_horiz(res):
        return {"rmse": np.mean([res[h]["rmse"] for h in res]),
                "mae": np.mean([res[h]["mae"] for h in res]),
                "mape": np.mean([res[h]["mape"] for h in res])}

    print("\n=== Summary (mean across selected horizons) ===")
    for name, res in [("Baseline", baseline_res), ("STACKED LSTM", lstm_res)]:
        m = mean_over_horiz(res)
        print(f"{name:12s} | Mean_RMSE: {m['rmse']:7.4f} | Mean_MAE: {m['mae']:7.4f} | Mean_MAPE: {m['mape']:7.2f}")

    print("\nDone. Models ->", MODEL_DIR, "Predictions ->", PRED_DIR)

if __name__ == "__main__":
    main()
