import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.signal import argrelextrema

# Optional: Set UTF-8 encoding for Windows console
if os.name == 'nt':
    sys.stdout.reconfigure(encoding='utf-8')

# Suppress TensorFlow verbose logging
tf.get_logger().setLevel('ERROR')

# Disable oneDNN logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class LRTrading:

    def __init__(self, stock, start, end):
        self.stock = stock
        self.start = start
        self.end = end
        self.scaler = MinMaxScaler()
        self.model = LogisticRegression()
        self.main_df = self.fetch_data()
        self.train_model()

    # Fetch stock data using yfinance
    def fetch_data(self):
        stock_data = yf.download(self.stock, self.start, self.end)
        stock_data['Normalized_Price'] = (stock_data['Close'] - stock_data['Low']) / (stock_data['High'] - stock_data['Low'])
        stock_data['Volume'] = stock_data['Volume']

        # Calculate rolling regression coefficients
        stock_data['3_day_reg'] = self.calculate_regression(stock_data, 3)
        stock_data['5_day_reg'] = self.calculate_regression(stock_data, 5)
        stock_data['10_day_reg'] = self.calculate_regression(stock_data, 10)
        stock_data['20_day_reg'] = self.calculate_regression(stock_data, 20)

        # Determine local maxima and minima
        max_indices = argrelextrema(stock_data['Close'].values, np.greater, order=5)[0]
        min_indices = argrelextrema(stock_data['Close'].values, np.less, order=5)[0]

        labels = np.full(len(stock_data), np.NaN)
        labels[max_indices] = 1  # Local maximum (sell signal)
        labels[min_indices] = 0   # Local minimum (buy signal)

        stock_data['Label'] = labels
        stock_data.dropna(subset=['Label'], inplace=True)

        return stock_data

    # Function to calculate linear regression coefficients over a rolling window
    def calculate_regression(self, stock_data, days):
        reg_coef = []
        for i in range(len(stock_data)):
            if i >= days:
                X = np.arange(days).reshape(-1, 1)
                y = stock_data['Close'].values[i-days:i]
                model = LinearRegression()
                model.fit(X, y)
                reg_coef.append(model.coef_[0])
            else:
                reg_coef.append(0)
        return reg_coef

    # Train the logistic regression model
    def train_model(self):
        x = self.main_df[['Normalized_Price', 'Volume', '3_day_reg', '5_day_reg', '10_day_reg', '20_day_reg']]
        y = self.main_df['Label'].values

        x_scaled = self.scaler.fit_transform(x)
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

        self.model.fit(self.train_x, self.train_y)

        # Evaluate the model
        self.evaluate_model()

    # Evaluate the model using confusion matrix
    def evaluate_model(self):
        predictions = self.model.predict(self.test_x)
        cm = confusion_matrix(self.test_y, predictions)
        cmd = ConfusionMatrixDisplay(cm)
        cmd.plot()
        plt.title(f"Confusion Matrix for {self.stock}")
        plt.show()
        print(f'Model accuracy: {self.model.score(self.test_x, self.test_y)}')

    # Test the model on new data and generate buy/sell signals
    def test_model(self, test_start, test_end):
        test_data = yf.download(self.stock, test_start, test_end)
        test_data['Normalized_Price'] = (test_data['Close'] - test_data['Low']) / (test_data['High'] - test_data['Low'])
        test_data['3_day_reg'] = self.calculate_regression(test_data, 3)
        test_data['5_day_reg'] = self.calculate_regression(test_data, 5)
        test_data['10_day_reg'] = self.calculate_regression(test_data, 10)
        test_data['20_day_reg'] = self.calculate_regression(test_data, 20)

        model_inputs = test_data[['Normalized_Price', 'Volume', '3_day_reg', '5_day_reg', '10_day_reg', '20_day_reg']].values
        model_inputs_scaled = self.scaler.transform(model_inputs)

        predicted_labels = self.model.predict(model_inputs_scaled)

        buy_signal, sell_signal = [None] * len(test_data), [None] * len(test_data)
        for i in range(len(predicted_labels)):
            if predicted_labels[i] == 0:
                buy_signal[i] = test_data['Close'].values[i]
            elif predicted_labels[i] == 1:
                sell_signal[i] = test_data['Close'].values[i]

        return test_data, buy_signal, sell_signal

    # Plot buy/sell signals
    def plot_signals(self, test_data, buy_signal, sell_signal):
        plt.figure(figsize=(12, 6))
        plt.plot(test_data['Close'].values, label="Stock Price", color='blue')
        plt.scatter(range(len(test_data['Close'])), buy_signal, color='green', label="Buy Signal", marker="^", alpha=1)
        plt.scatter(range(len(test_data['Close'])), sell_signal, color='red', label="Sell Signal", marker="v", alpha=1)
        plt.title(f"{self.stock} Stock Price with Buy/Sell Signals")
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()

# Example of running the model
if True:
    lr_model = LRTrading(stock='SPY', start='2005-01-01', end='2020-01-01')
    test_data, buy_signal, sell_signal = lr_model.test_model(test_start='2020-01-01', test_end='2021-01-01')
    lr_model.plot_signals(test_data, buy_signal, sell_signal)
