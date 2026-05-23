from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib

# Fix Render/Flask graph issue
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

import os
import time

app = Flask(__name__)

# Create folders automatically
os.makedirs("stocks", exist_ok=True)
os.makedirs("static", exist_ok=True)


# ==========================================
# HOME PAGE
# ==========================================

@app.route('/')
def home():
    return render_template("index.html")


# ==========================================
# PREDICTION ROUTE
# ==========================================

@app.route('/predict', methods=['POST'])
def predict():

    stock = request.form['stock'].upper().strip()

    # ==========================================
    # FETCH LIVE DATA
    # ==========================================

    try:
        time.sleep(1)

        data = yf.download(
            stock,
            period='5y',
            auto_adjust=True,
            progress=False,
            threads=False
        )

        # ==========================================
        # AUTO UPDATE CSV
        # ==========================================
        if not data.empty:
            csv_path = f"stocks/{stock}.csv"
            data.to_csv(csv_path)
            print(f"{stock} CSV updated successfully.")
        else:
            print("Yahoo returned empty data. Using CSV fallback...")

            csv_path = f"stocks/{stock}.csv"

            if os.path.exists(csv_path):
                data = pd.read_csv(csv_path)
                data['Date'] = pd.to_datetime(data['Date'])
                data.set_index('Date', inplace=True)
                data.sort_index(inplace=True)
                print("CSV fallback loaded.")
            else:
                return render_template(
                    "index.html",
                    error="❌ Stock data unavailable."
                )

        # Fix MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

    except Exception as e:
        print("Yahoo Error:", e)

        csv_path = f"stocks/{stock}.csv"

        if os.path.exists(csv_path):
            data = pd.read_csv(csv_path)
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            data.sort_index(inplace=True)
            print("CSV fallback loaded.")
        else:
            return render_template(
                "index.html",
                error="❌ Yahoo blocked requests and no CSV backup found."
            )

    # ==========================================
    # DATA CLEANING
    # ==========================================

    if 'close' in data.columns:
        data.rename(columns={'close': 'Close'}, inplace=True)

    if 'Close' not in data.columns:

        return render_template(
            "index.html",
            error="❌ Close price data missing."
        )

    data = data[['Close']]

    data['Close'] = pd.to_numeric(
        data['Close'],
        errors='coerce'
    )

    data.dropna(inplace=True)

    if data.empty:

        return render_template(
            "index.html",
            error="❌ No valid stock data available."
        )

    # ==========================================
    # FEATURE ENGINEERING
    # ==========================================

    data['Prev_Close'] = data['Close'].shift(1)

    data['MA5'] = data['Close'].rolling(window=5).mean()

    data['MA10'] = data['Close'].rolling(window=10).mean()

    data['Daily_Return'] = data['Close'].pct_change()

    data['Volatility'] = (
        data['Daily_Return']
        .rolling(window=5)
        .std()
    )

    # Target column
    data['Prediction'] = data['Close'].shift(-1)

    # Remove NaN rows
    data.dropna(inplace=True)

    # ==========================================
    # MACHINE LEARNING MODEL
    # ==========================================

    X = data[[
        'Prev_Close',
        'MA5',
        'MA10',
        'Daily_Return',
        'Volatility'
    ]]

    y = data['Prediction']

    # Random Forest Model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    # Latest data
    latest_data = X.tail(1)

    # Prediction
    prediction = model.predict(latest_data)

    predicted_price = round(
        float(prediction[0]),
        2
    )

    print("\n========================")
    print("LATEST STOCK:", stock)
    print("LATEST DATE:", data.index[-1])
    print("LATEST CLOSE:", data['Close'].iloc[-1])
    print("PREDICTED PRICE:", predicted_price)
    print("========================\n")

    # ==========================================
    # GRAPH
    # ==========================================

    plt.figure(figsize=(12, 6))

    plot_df = data.copy()

    plot_df.sort_index(inplace=True)

    # Only latest prediction point
    plot_df['Predicted'] = np.nan

    plot_df.at[
        plot_df.index[-1],
        'Predicted'
    ] = predicted_price

    # Actual price
    plt.plot(
        plot_df.tail(60).index,
        plot_df['Close'].tail(60),
        label='Actual Price',
        color='blue',
        linewidth=2
    )

    # Predicted point
    plt.scatter(
        plot_df.tail(60).index[-1],
        predicted_price,
        color='red',
        s=100,
        label='Next Day Prediction'
    )

    # Labels
    plt.title(f"{stock} Stock Prediction")

    plt.xlabel("Date")

    plt.ylabel("Price")

    plt.legend()

    plt.grid(True)

    import matplotlib.dates as mdates

    plt.gca().xaxis.set_major_formatter(
        mdates.DateFormatter('%Y-%m')
    )

    plt.xticks(rotation=30)

    plt.tight_layout()

    # Unique graph name to avoid caching
    graph_path = f"static/{stock}_graph.png"

    plt.savefig(graph_path)

    plt.close()

    # ==========================================
    # RETURN RESULT
    # ==========================================

    return render_template(
        "index.html",
        prediction=predicted_price,
        stock=stock,
        graph=graph_path
    )


# ==========================================
# RUN APP
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)