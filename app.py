from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
import time

app = Flask(__name__)

# Create stocks folder automatically
os.makedirs("stocks", exist_ok=True)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    stock = request.form['stock'].upper().strip()

    # ==========================================
    # FETCH LIVE DATA + CSV FALLBACK
    # ==========================================

    try:

        time.sleep(2)

        data = yf.download(
            stock,
            period='5y',
            auto_adjust=True,
            progress=False,
            threads=False
        )

        # Fix MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # If Yahoo fails → CSV fallback
        if data.empty:

            print("Yahoo failed. Using CSV fallback...")

            csv_path = f"stocks/{stock}.csv"

            if os.path.exists(csv_path):

                data = pd.read_csv(csv_path)

                print("CSV fallback loaded.")

            else:

                return render_template(
                    "index.html",
                    error="❌ Stock data unavailable."
                )

    except Exception as e:

        print("Yahoo Error:", e)

        csv_path = f"stocks/{stock}.csv"

        if os.path.exists(csv_path):

            data = pd.read_csv(csv_path)

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

    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')

    data = data.dropna()

    if data.empty:

        return render_template(
            "index.html",
            error="❌ No valid stock data available."
        )

    # ==========================================
    # MACHINE LEARNING
    # ==========================================

    forecast_days = 1

    data['Prediction'] = data[['Close']].shift(-forecast_days)

    data = data.dropna()

    X = np.array(data[['Close']])

    y = np.array(data['Prediction'])

    model = LinearRegression()

    model.fit(X, y)

    latest_price = np.array(data[['Close']].tail(1))

    prediction = model.predict(latest_price)

    predicted_price = round(prediction[0], 2)
    # ==========================================
    # GRAPH
    # ==========================================

    # Ensure static folder exists
    os.makedirs("static", exist_ok=True)

    plt.figure(figsize=(12, 6))

    # Create plotting dataframe from processed `data`
    plot_df = data.copy()
    plot_df.sort_index(inplace=True)

    # Set latest predicted value on the newest row
    plot_df['Prediction'] = plot_df['Close'].shift(-1)
    plot_df.at[plot_df.index[-1], 'Prediction'] = predicted_price

    # Actual stock prices
    plt.plot(
        plot_df.tail(60).index,
        plot_df['Close'].tail(60),
        label='Actual Price',
        color='blue',
        linewidth=2,
    )

    # Predicted prices
    plt.plot(
        plot_df.tail(60).index,
        plot_df['Prediction'].tail(60),
        linestyle='--',
        label='Predicted Price',
        color='red',
        linewidth=2,
    )

    # Graph labels
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

    graph_path = os.path.join("static", "graph.png")
    plt.savefig(graph_path)
    plt.close()

    return render_template(
        "index.html",
        prediction=predicted_price,
        stock=stock,
        graph=graph_path
    )

if __name__ == "__main__":
    app.run(debug=True)