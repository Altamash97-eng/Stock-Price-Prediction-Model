from flask import Flask,render_template,request
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import joblib

app = Flask(__name__)

# Load trained model
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

    # Get stock symbol from form
    stock = request.form['stock']

    # Download latest stock data
    data = yf.download(
     tickers=stock,
     period="1y",
     interval="1d",
     progress=False,
     auto_adjust=True,
     threads=False
    )

    print(data.head())
    print("Rows:", len(data))
    if data.empty:
     return render_template(
        "index.html",
        prediction_text="Invalid stock symbol or no data found"
    )

    # Keep only Close prices
    data = data[['Close']]

    # Create prediction column
    data['Prediction'] = data[['Close']].shift(-1)
    data.dropna(inplace=True)

    # Features and labels
    X = np.array(data.drop(['Prediction'], axis=1))[:-1]

    y = np.array(data['Prediction'])[:-1]

    # Train model
    model.fit(X, y)

    # Latest price
    last_price = np.array(data[['Close']].tail(1))

    # Predict next day price
    next_prediction = model.predict(last_price)

    predicted_price = round(float(next_prediction[0]), 2)
   

    # =========================
    # CREATE GRAPH
    # =========================

    graph_data = data.tail(100).copy()

    graph_data['Predicted'] = graph_data['Close'].shift(-1)

    graph_data.dropna(inplace=True)

    # Graph figure
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

    # Return to HTML page
    return render_template(

        "index.html",

        prediction_text=f"Predicted Price: ${predicted_price}",

        stock=stock

    )

# =========================
# RUN APP
# =========================

import os

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )