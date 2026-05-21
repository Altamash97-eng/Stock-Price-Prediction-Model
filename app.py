from flask import Flask
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

        stock,

        start='2020-01-01',

        end='2026-05-01'

    )

    # Keep only Close prices
    data = data[['Close']]

    # Create prediction column
    data['Prediction'] = data[['Close']].shift(-1)

    # Features and labels
    X = np.array(data.drop(['Prediction'], axis=1))[:-1]

    y = np.array(data['Prediction'])[:-1]

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

        'index.html',

        prediction=predicted_price,

        stock=stock

    )

# =========================
# RUN APP
# =========================

if __name__ == '__main__':

    app.run(host='0.0.0.0',port=5000)