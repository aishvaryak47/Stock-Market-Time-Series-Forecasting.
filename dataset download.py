import requests
import pandas as pd

# Your API key
api_key = "MjT1uViB5fsHHNqwNTlaVPFOKtoVGigv"

# FMP API endpoint for historical stock data
url = "https://financialmodelingprep.com/api/v3/historical-price-full/RELIANCE.NS"

# Parameters for the request
params = {
    "apikey": api_key,
    "from": "2010-01-01",
    "to": "2025-08-19"
}

# Send GET request
response = requests.get(url, params=params)
data = response.json()

# Safely get the historical data
historical_data = data.get('historical') or data.get('historicalStockList')

if not historical_data:
    raise ValueError("No historical data found! Check your API key or parameters.")

# Convert to DataFrame
df = pd.DataFrame(historical_data)

# Save CSV in your src folder
csv_path = "C:\\Users\\selco\\OneDrive\\Desktop\\stock_market\\src\\RELIANCE.NS.csv"
df.to_csv(csv_path, index=False)

print(f"Data downloaded and saved to {csv_path}")
