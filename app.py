from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
import time

# For Render deployment
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import joblib

app = Flask(__name__)

# Load trained ML model
model = joblib.load('stock_model.pkl')

# =========================
# HOME PAGE
# =========================

@app.route('/')
def home():

    return render_template('index.html')

# =========================
# PREDICTION ROUTE
# =========================

@app.route('/predict', methods=['POST'])
def predict():

    # Get stock symbol from HTML form
    stock = request.form['stock']


    # Download stock data
    try:
        data = yf.download(
            stock,
            period='5y',
            auto_adjust=True,
            progress=False,
            threads=False
        )

        # Fix MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Retry if empty
        if data.empty:
            time.sleep(2)
            data = yf.download(
                stock,
                period='5y',
                auto_adjust=True,
                progress=False,
                threads=False
            )

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

    except Exception:
        return render_template(
            "index.html",
            error="❌ Yahoo Finance temporary limit reached. Try again later."
        )

    # Check if stock data exists

    if data.empty:

      return render_template(

        'index.html',

        error="❌ Stock data not found. Please enter a valid stock symbol."

    )
    # WRITE HERE 
    # Remove NaN rows
    data = data.dropna()

    # Check again after removing NaN
    if data.empty:
     return render_template(
        'index.html',
        error="❌ Stock data unavailable for prediction."
    )

    # Keep only Close prices
    data = data[['Close']]
    #Remove NaN again
    data = data.dropna()

    # Create next day prediction column
    data['Prediction'] = data[['Close']].shift(-1)

    # Features and labels
    X = np.array(data.drop(['Prediction'], axis=1))[:-1]

    y = np.array(data['Prediction'])[:-1]

    # Latest stock price
    last_price = np.array(data[['Close']].tail(1))

    # Predict next day stock price
    next_prediction = model.predict(last_price)

    predicted_price = round(float(next_prediction[0]), 2)

    # =========================
    # CREATE GRAPH
    # =========================

    graph_data = data.tail(100).copy()

    graph_data['Predicted'] = graph_data['Close'].shift(-1)

    graph_data.dropna(inplace=True)

    # Create graph figure
    plt.figure(figsize=(12,6))

    # Actual prices
    plt.plot(

        graph_data.index,

        graph_data['Close'],

        label='Actual Price',

        color='blue'

    )

    # Predicted prices
    plt.plot(

        graph_data.index,

        graph_data['Predicted'],

        label='Predicted Price',

        color='red',

        linestyle='dashed'

    )

    # Graph details
    plt.title(f"{stock} Stock Prediction")

    plt.xlabel("Date")

    plt.ylabel("Price")

    plt.legend()

    plt.grid(True)

    # Save graph
    plt.savefig('static/graph.png')

    plt.close()

    # Return result to HTML
    return render_template(

        'index.html',

        prediction=predicted_price,

        stock=stock

    )

# =========================
# RUN FLASK APP
# =========================

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000)