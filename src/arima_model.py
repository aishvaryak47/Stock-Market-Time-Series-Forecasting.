import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# 1️⃣ Load and clean CSV
df = pd.read_csv("RELIANCE.NS.csv", skiprows=2)
df = df.rename(columns={df.columns[0]: 'Date', df.columns[1]: 'Close'})
df = df[['Date', 'Close']]
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df.set_index('Date', inplace=True)
series = df['Close']

# 2️⃣ Split into train/test
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

# 3️⃣ Fit ARIMA
model = ARIMA(train, order=(5,1,0))
model_fit = model.fit()

# 4️⃣ Forecast for test set
forecast = model_fit.forecast(steps=len(test))
mse = mean_squared_error(test, forecast)
print(f"Mean Squared Error: {mse}")

# 5️⃣ Forecast next 30 days (future)
future_forecast = model_fit.forecast(steps=30)
print("Next 30 Days Forecast:")
print(future_forecast)

# 6️⃣ Plot historical + future forecast **<<--- THIS IS WHERE YOU ADD IT**
plt.figure(figsize=(12,6))
plt.plot(df.index, df['Close'], label='Historical')
plt.plot(pd.date_range(df.index[-1]+pd.Timedelta(1, unit='d'), periods=30),
         future_forecast, label='Next 30 Days Forecast', color='red')
plt.title("Reliance Stock Price Forecast")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()

# Save next 30 days forecast to CSV
future_forecast.to_csv("future_30_days_forecast.csv", header=True)
print("Future forecast saved as future_30_days_forecast.csv in src folder")


