import pandas as pd, matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load CSV with explicit date format
df = pd.read_csv(
    "data/RELIANCE.NS.csv",
    index_col=0,
    parse_dates=[0],
    date_format="%Y-%m-%d"
)

# Ensure Close is numeric
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df = df.dropna(subset=["Close"])

# Closing price trend
df["Close"].plot(title="Closing Price", figsize=(10,5))
plt.savefig("outputs/figures/closing.png")

# Decomposition (trend, seasonality, residuals)
result = seasonal_decompose(df["Close"], model="additive", period=30)
result.plot()
plt.savefig("outputs/figures/decomposition.png")
