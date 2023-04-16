import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from arch.unitroot.cointegration import DynamicOLS
import requests
from io import StringIO


def IMPORTTWELVEDATA(ticket,interval,size):

#Supported intervals: 1min, 5min, 15min, 30min, 45min, 1h, 2h, 4h, 8h, 1day, 1week, 1month
    url = "https://twelve-data1.p.rapidapi.com/time_series"

    querystring = {"symbol":ticket,"interval":interval,"outputsize":size,"format":"csv"}

    headers = {
        'x-rapidapi-key': "d48890a58emsh5e7affca7714a71p1ab6e1jsn4ba14ffa0b6c",
        'x-rapidapi-host': "twelve-data1.p.rapidapi.com"
        }

    response = requests.request("GET", url, headers=headers, params=querystring)

    TESTDATA = StringIO(response.text)
    df = pd.read_csv(TESTDATA, sep=";")
    df=df.assign(datetime=pd.to_datetime(df['datetime']))
    df=df.sort_values(by='datetime')
    df = df.reset_index().drop('index', axis = 1)
    
    return df

def test_cointeg(tick1, tick2, interval, size):
    
    #get data
    df_btc=IMPORTTWELVEDATA(tick1,interval, size)
    df_eth=IMPORTTWELVEDATA(tick2,interval, size)
    btc = [(df_btc['close'][i]-np.mean(df_btc['close']))/np.std(df_btc['close']) for i in range(size)]
    eth = [(df_eth['close'][i]-np.mean(df_eth['close']))/np.std(df_eth['close']) for i in range(size)]
    df_btc['cr_close']=btc
    df_eth['cr_close']=eth
    full_df=df_btc[['datetime','close', 'cr_close']]
    full_df=full_df.rename(columns={'close':'btc_close', 'cr_close':'btc_cr_close'})
    full_df['eth_close']=df_eth['close']
    full_df['eth_cr_close']=df_eth['cr_close']
    

    #model
    model=DynamicOLS(df_btc['cr_close'],df_eth['cr_close'])
    res=model.fit()
    print(res.summary())

    # Get the residuals from the DynamicOLS model
    residuals = res.resid

    # Perform the ADF test on the residuals
    adf_test = adfuller(residuals)

    # Print the test results
    print("ADF Statistic: %f" % adf_test[0])
    print("p-value: %f" % adf_test[1])
    print("Critical Values:")
    for key, value in adf_test[4].items():
        print("\t%s: %.3f" % (key, value))
        
    if adf_test[1]<0.05:
        cointeg=True
        print('residual cointegration OK')
        mispricing_ptf = df_btc['cr_close'] - df_eth['cr_close']
        full_df['misp_ptf']=mispricing_ptf
    else: 
        print('NO residual cointegration')
        cointeg=False
        mispricing_ptf = df_btc['cr_close'] - df_eth['cr_close']
        full_df['misp_ptf']=mispricing_ptf
        
    adf_test2 = adfuller(mispricing_ptf)
    print("ADF Statistic: %f" % adf_test2[0])
    print("p-value: %f" % adf_test2[1])
    print("Critical Values:")
    for key, value in adf_test2[4].items():
        print("\t%s: %.3f" % (key, value))

    plt.savefig('static/graphique.png')
    return full_df, cointeg

