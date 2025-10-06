import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# 1️⃣ Load and preprocess CSV
df = pd.read_csv("RELIANCE.NS.csv", skiprows=2)
df = df.rename(columns={df.columns[0]: 'ds', df.columns[1]: 'y'})
df = df[['ds', 'y']]
df['ds'] = pd.to_datetime(df['ds'])
df = df.sort_values('ds').reset_index(drop=True)

# 2️⃣ Initialize and fit Prophet model
prophet_model = Prophet(daily_seasonality=True)
prophet_model.fit(df)

# 3️⃣ Create future dataframe for 30 days
future = prophet_model.make_future_dataframe(periods=30)
forecast = prophet_model.predict(future)

# 4️⃣ Save forecast CSV
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv("future_30_days_prophet_forecast.csv", index=False)
print("Prophet forecast saved as future_30_days_prophet_forecast.csv")

# 5️⃣ Plot historical + forecast
prophet_model.plot(forecast)
plt.title("Reliance Stock Price Prophet Forecast")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.savefig("Prophet_forecast_plot.png")
plt.show()
print("Prophet forecast plot saved as Prophet_forecast_plot.png")
