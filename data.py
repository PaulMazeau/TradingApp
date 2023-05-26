import pandas as pd
import requests
from io import StringIO
import yfinance as yf
from pandas_datareader import data as web
import warnings
warnings.simplefilter('ignore')

class GetData:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    def dl_close_data(self, list_tickers, full_tickers):
        print(f'Get market data for {len(full_tickers)} tickers')
        res = {}
        for tick in list_tickers:
            yf.pdr_override()
            df = web.get_data_yahoo(tick, start=self.start_date, end=self.end_date, progress=False)
            df['ticker'] = tick
            res[tick] = df

        data = pd.concat(res)
        data.reset_index(inplace=True)
        data = data.pivot(index='Date', columns='ticker', values = 'Adj Close')
        return data

    def dl_full_data(self, list_tickers, full_tickers):
        print(f'Get market data for {len(full_tickers)} tickers')
        res = {}
        for tick in list_tickers:
            yf.pdr_override()
            df = web.get_data_yahoo(tick, start=self.start_date, end=self.end_date, progress=False)
            df['ticker'] = tick
            res[tick] = df

        data = pd.concat(res)
        data.reset_index(inplace=True)
        return data

    @staticmethod
    def twelve_data(list_tickers, interval):
        url = "https://twelve-data1.p.rapidapi.com/time_series"
        full_df = pd.DataFrame(columns=list_tickers, index=range(5000))
        for ticket in list_tickers:
            querystring = {"symbol": ticket, "interval": interval, "outputsize": 5000, "format": "csv"}

            headers = {
                'x-rapidapi-key': "d48890a58emsh5e7affca7714a71p1ab6e1jsn4ba14ffa0b6c",
                'x-rapidapi-host': "twelve-data1.p.rapidapi.com"
            }

            response = requests.request("GET", url, headers=headers, params=querystring)

            test_data = StringIO(response.text)
            df = pd.read_csv(test_data, sep=";")
            df = df.assign(datetime=pd.to_datetime(df['datetime']))
            df = df.sort_values(by='datetime')
            df = df.reset_index().drop('index', axis=1)
            full_df[ticket] = df['close']
        # data.reset_index(inplace=True)
        return full_df