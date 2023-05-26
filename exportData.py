import alpaca_trade_api as tradeapi
from alpaca.trading.client import TradingClient
from alpaca_trade_api.rest import REST, TimeFrame
import pandas as pd

api = tradeapi.REST('PKDO7AS76AVP98TH2VHN', 'kUFaGiITTNBIAeHilNNcUpVaCNffHfTzgktV3Mcd', 'https://paper-api.alpaca.markets')
trading_client = TradingClient('PKDO7AS76AVP98TH2VHN', 'kUFaGiITTNBIAeHilNNcUpVaCNffHfTzgktV3Mcd', paper=True)


def exportData(tck1, startDate, endDate):
    df = api.get_bars(tck1, TimeFrame.Day, startDate, endDate, adjustment='raw',).df
    # Convert the start and end dates to datetime objects
    start_date = pd.to_datetime(startDate)
    end_date = pd.to_datetime(endDate)
    # Calculate the difference in days between the two dates
    duration = (end_date - start_date).days
    df.to_csv(f'{tck1}_{duration}_days.csv')


