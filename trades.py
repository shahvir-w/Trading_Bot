from alpaca.trading.client import TradingClient
from alpaca.data import StockHistoricalDataClient, StockTradesRequest
from datetime import datetime

from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from alpaca.data.live import StockDataStream

from alpaca_trade_api import REST
from timedelta import Timedelta

API_KEY = 'PKIPIRLQAHOYUI480SV7'
SECRET_KEY = 'VIwqZtuxlo1h6hgf1i4ZgH3hFHkz06ePqfdxnpwb'

trading_client = TradingClient(API_KEY, SECRET_KEY)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

request_params = StockTradesRequest(
    symbol_or_symbols="AAPL",
    start=datetime(2024, 9, 30, 14, 30),
    end=datetime(2024, 9, 30, 14, 45)
)
trades = data_client.get_stock_trades(request_params)

market_order_data = MarketOrderRequest(
    symbol="SPY",
    qty=1,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.DAY
)
#market_order = trading_client.submit_order(market_order_data)

stream = StockDataStream(API_KEY, SECRET_KEY)

async def handle_trade(data):
    print(data)

stream.subscribe_trades(handle_trade, "AAPL")
stream.run()