import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# 1️⃣ Load and preprocess CSV
df = pd.read_csv("RELIANCE.NS.csv", skiprows=2)
df = df.rename(columns={df.columns[0]: 'Date', df.columns[1]: 'Close'})
df = df[['Date', 'Close']]
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df.set_index('Date', inplace=True)

# 2️⃣ Split into train/test
series = df['Close']
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

# 3️⃣ Fit SARIMA model
sarima_model = SARIMAX(train, 
                       order=(5,1,0),        # p,d,q
                       seasonal_order=(1,1,1,12))  # P,D,Q,s ; s=12 for monthly seasonality
sarima_fit = sarima_model.fit(disp=False)

# 4️⃣ Forecast on test set
sarima_forecast = sarima_fit.forecast(steps=len(test))
mse_sarima = mean_squared_error(test, sarima_forecast)
print(f"SARIMA Mean Squared Error: {mse_sarima}")

# 5️⃣ Forecast next 30 days
future_sarima_forecast = sarima_fit.forecast(steps=30)
print("SARIMA Next 30 Days Forecast:")
print(future_sarima_forecast)

# Save forecast CSV
future_sarima_forecast.to_csv("future_30_days_sarima_forecast.csv", header=True)
print("Future SARIMA forecast saved as future_30_days_sarima_forecast.csv")

# 6️⃣ Plot historical + forecast
plt.figure(figsize=(12,6))
plt.plot(df.index, df['Close'], label='Historical')
plt.plot(pd.date_range(df.index[-1]+pd.Timedelta(1, unit='d'), periods=30),
         future_sarima_forecast, label='SARIMA Next 30 Days Forecast', color='green')
plt.title("Reliance Stock Price SARIMA Forecast")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.savefig("SARIMA_forecast_plot.png")
plt.show()
print("SARIMA forecast plot saved as SARIMA_forecast_plot.png")
future_sarima_forecast.to_csv("future_30_days_sarima_forecast.csv", header=True)
plt.savefig("SARIMA_forecast_plot.png")


