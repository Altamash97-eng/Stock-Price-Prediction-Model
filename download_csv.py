import yfinance as yf
import pandas as pd
import os

# STOCKS TO DOWNLOAD CSV
stocks = [

    # US Stocks
    "AAPL",
    "TSLA",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",

    # Indian Stocks
    "TCS.NS",
    "INFY.NS",
    "RELIANCE.NS",
    "SBIN.NS",
    "HDFCBANK.NS",
    "ITC.NS",
    "WIPRO.NS"
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "APOLLOHOSP.NS",
    "ASIANPAINT.NS",
    "AXISBANK.NS",
    "BAJAJ-AUTO.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "BEL.NS",
    "BHARTIARTL.NS",
    "BPCL.NS",
    "BRITANNIA.NS",
    "CIPLA.NS",
    "COALINDIA.NS",
    "DRREDDY.NS",
    "EICHERMOT.NS",
    "ETERNAL.NS",
    "GRASIM.NS",
    "HCLTECH.NS",
    "HDFCLIFE.NS",
    "HEROMOTOCO.NS",
    "HINDALCO.NS",
    "HINDUNILVR.NS",
    "ICICIBANK.NS",
    "INDUSINDBK.NS",
    "JIOFIN.NS",
    "JSWSTEEL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "M&M.NS",
    "MARUTI.NS",
    "NESTLEIND.NS",
    "NTPC.NS",
    "ONGC.NS",
    "POWERGRID.NS",
    "SBILIFE.NS",
    "SHRIRAMFIN.NS",
    "SUNPHARMA.NS",
    "TATACONSUM.NS",
    "TATAMOTORS.NS",
    "TATASTEEL.NS",
    "TECHM.NS",
    "TITAN.NS",
    "TRENT.NS",
    "ULTRACEMCO.NS",
]

# Create stocks folder
os.makedirs("stocks", exist_ok=True)

# Download CSV files
for stock in stocks:

    print(f"Downloading {stock}...")

    try:

        data = yf.download(
            stock,
            period="5y",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        # Fix MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if not data.empty:

            data.to_csv(f"stocks/{stock}.csv")

            print(f"{stock} CSV saved.")

        else:

            print(f"{stock} returned empty data.")

    except Exception as e:

        print(f"Error downloading {stock}: {e}")

print("All CSV downloads completed.")