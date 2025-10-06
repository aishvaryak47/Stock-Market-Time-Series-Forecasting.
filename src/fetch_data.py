import yfinance as yf
import sys

ticker = sys.argv[1]   # example: RELIANCE.NS
data = yf.download(ticker, start="2018-01-01", end="2024-12-31")
data.to_csv(f"data/{ticker}.csv")
print("Saved data/", ticker, ".csv")
