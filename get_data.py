import yfinance as yf
from datetime import datetime
import pandas as pd

end_date = datetime.today().strftime('%Y-%m-%d')

ticker_symbol = 'AAPL'
stock_data = yf.download(ticker_symbol, start='2020-01-01', end=end_date)

print(stock_data.isnull().sum())

#print(stock_data.tail())