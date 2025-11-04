# 🚦 Traffic Flow Prediction using LSTM (METR-LA Dataset)
Developed by **Sachin S** 

---

## 📌 Abstract

This project evaluates an LSTM-based deep learning model for traffic prediction. The model is compared against a historical-average baseline using RMSE, MAE, and MAPE metrics at 15, 30, and 60-minute prediction horizons. Results show the LSTM outperforms the baseline in RMSE and MAPE, confirming its ability to learn temporal dependencies in multi-sensor traffic data.

---

## 📂 Dataset Details

| Property | Value |
|----------|--------|
| Dataset | METR-LA (Public Benchmark) |
| Resolution | 5-minute interval |

---

## 🛠️ Technology Stack

| Layer | Tools / Libraries |
|-------|-------------------|
| Programming Language | Python 3.10+ |
| Deep Learning | TensorFlow / Keras |
| Data Handling | NumPy, Pandas, h5py |
| Evaluation Metrics | RMSE, MAE, MAPE (scikit-learn + custom) |
| Scaling / Preprocessing | MinMaxScaler (scikit-learn) |
| Dataset Format | METR-LA `.h5` traffic sensor dataset |
| Environment | Jupyter / VS Code |

---

## Methodology

### Task:
Build an ML model to predict short-term traffic flow (e.g., vehicle count or speed) for a set of sensors/road segments for the next 15–60 minutes given historical sensor data.

### Model Architecture (Why LSTM?)

A stacked LSTM network was selected because:

- Traffic data is highly temporal and sequential, with autocorrelation and rush-hour patterns.  
- LSTM is designed to retain long-term dependencies, unlike simple feed-forward models.  
- Works well in multivariate sensor-based forecasting where each timestep has many features (207 sensors).  
- Lightweight enough to run on CPU if GPU unavailable.  

---

## Architecture Overview (Stacked LSTM):  
- `Input: (12 timesteps × 207 sensors)`  
- `LSTM (128 units)`  
- `Dropout (0.3)`  
- `Dense (12 × 207 units)`  
- `Reshape → (12 timesteps × 207 sensors)`  
- **Loss:** MSE  
- **Optimizer:** Adam (lr = 5e-4)

✅ Handles multivariate time-series (all sensors at once)  
✅ Captures temporal patterns  

---

## ⚙️ Training Setup

| Item | Value |
|------|--------|
| Lookback window | 12 steps (1 hour) |
| Forecast horizon | 12 steps (1 hour) |
| Train/Val/Test split | 70 / 10 / 20 |
| Epochs | 10 |
| Batch size | 64 |
| Scaler | Min-Max |

---

## 📊 Results

### 🔹 Baseline: Historical Average
| Horizon | Minutes | RMSE | MAE | MAPE (%) |
|---------|----------|------|-----|----------|
| 3  | 15 | 12.18 | 5.52 | 12.29 |
| 6  | 30 | 14.11 | 6.56 | 14.61 |
| 12 | 60 | 16.86 | 8.30 | 18.55 |

### 🔹 Proposed Model: LSTM
| Horizon | Minutes | RMSE | MAE | MAPE (%) |
|---------|----------|------|-----|----------|
| 3  | 15 | 11.94 | 6.61 | 12.62 |
| 6  | 30 | 13.47 | 7.44 | 13.77 |
| 12 | 60 | 15.47 | 8.75 | 15.86 |

### 📌 Mean Performance
| Model | Mean RMSE | Mean MAE | Mean MAPE |
|-------|-----------|----------|-----------|
| Baseline | 14.39 | 6.79 | 15.15 |
| **LSTM** | **13.63** | 7.60 | **14.08** |

✅ Lower RMSE and MAPE than baseline  

---

## 🧩 Interpretation

| Strengths | Weaknesses |
|-----------|------------|
| Learns temporal dependencies | No explicit spatial modeling |
| Better long-horizon forecasting | MAE gap vs baseline |
| Handles all 207 sensors jointly | Accuracy drops with horizon |

---

## 🚀 Future Improvements

| Idea | Reason |
|------|--------|
| Add Graph Neural Networks (DCRNN, ST-GCN) | Traffic sensors are spatially connected |
| Seq2Seq / Transformer decoder | Better multi-step prediction |

---

## 📚 References

- Yu et al. (2018), *Spatio-Temporal Graph Convolutional Networks*
- Hochreiter & Schmidhuber (1997), *LSTM*
- METR-LA dataset (Caltrans PeMS)

---

## 👩‍💻 Author
**Sachin S**  
B.Tech – Artificial Intelligence and Data Science  
Rajalakshmi Institute of Technology, Chennai  
📧 svsachinsd@gmail.com

