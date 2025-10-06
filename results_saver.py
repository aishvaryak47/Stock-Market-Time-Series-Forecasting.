# results_saver.py
import pandas as pd

# Example: after evaluating models
results = {
    "Model": ["ARIMA", "LSTM", "RandomForest"],
    "MAE": [12.5, 8.3, 10.1],
    "MSE": [180.2, 120.4, 140.6],
    "RMSE": [13.4, 10.9, 11.8],
    "R2": [0.72, 0.85, 0.80]
}

df = pd.DataFrame(results)

# Save results to CSV (this will be used by Streamlit dashboard)
df.to_csv("model_results.csv", index=False)
print("✅ Model results saved to model_results.csv")
