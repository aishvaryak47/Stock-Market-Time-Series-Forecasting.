import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

# === Load CSV ===
df = pd.read_csv("data/RELIANCE.NS.csv", header=1, index_col=0, parse_dates=True)

# Rename columns properly
df.columns = ["Close", "High", "Low", "Open", "Volume"]

print("Data Preview:")
print(df.head())

# === Use only Close column ===
data = df[['Close']].values

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Train-test split
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# === Function to create sequences ===
def create_sequences(dataset, seq_length=60):
    X, y = [], []
    for i in range(len(dataset) - seq_length):
        X.append(dataset[i:i+seq_length, 0])
        y.append(dataset[i+seq_length, 0])
    return np.array(X), np.array(y)

# Prepare sequences
seq_length = 60
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Reshape for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# === Build LSTM model ===
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1)

# === Predictions ===
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# === Plot and Save ===
plt.figure(figsize=(14,5))
plt.plot(df.index[train_size+seq_length:], data[train_size+seq_length:], label="Actual Price")
plt.plot(df.index[train_size+seq_length:], predictions, label="Predicted Price")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()

# Save plot
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "reliance_lstm_prediction.png"))
plt.show()

# === Save predictions to CSV ===
pred_df = pd.DataFrame({
    "Date": df.index[train_size+seq_length:],
    "Actual": data[train_size+seq_length:].flatten(),
    "Predicted": predictions.flatten()
})
pred_df.to_csv(os.path.join(output_dir, "reliance_predictions.csv"), index=False)

print("✅ Plot saved in 'results/reliance_lstm_prediction.png'")
print("✅ Predictions saved in 'results/reliance_predictions.csv'")
