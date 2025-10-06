import yfinance as yf

# Download historical data for RELIANCE.NS
data = yf.download("RELIANCE.NS", start="2010-01-01", end="2025-08-19")

# Save to CSV
data.to_csv("RELIANCE.NS.csv")

print("Data downloaded and saved as RELIANCE.NS.csv")
