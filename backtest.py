import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from arch.unitroot.cointegration import DynamicOLS
import requests
from io import StringIO
from bot import test_cointeg

# real data
import requests
from io import StringIO


def IMPORTTWELVEDATA(ticket,interval,size):


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




#Supported intervals: 1min, 5min, 15min, 30min, 45min, 1h, 2h, 4h, 8h, 1day, 1week, 1month



def backtest(tick1,tick2,interval, size):
    df, cointeg=test_cointeg(tick1,tick2,interval, size)
    #if cointeg:
    #print('COINTEGRED')
    pos='nul'
    ptf=2000
    nb_eth=0
    nb_btc=0
    benef=[0]
    prix_eth=[]
    prix_btc=[]
    benef_eth=[]
    benef_btc=[]
    for i in range(2,df.shape[0]):
        if pos=='nul':
            if df['misp_ptf'].iloc[i]<-np.std(df['misp_ptf']) and df['misp_ptf'].iloc[i-1]>=-np.std(df['misp_ptf']):
                pos='short_eth'
                prix_eth.append(df['eth_close'].iloc[i])
                prix_btc.append(-df['btc_close'].iloc[i])
                nb_eth = (1000/df['eth_close'].iloc[i])
                nb_btc = (1000/df['btc_close'].iloc[i])
                ptf=0
            elif df['misp_ptf'].iloc[i]>np.std(df['misp_ptf']) and df['misp_ptf'].iloc[i-1]<=np.std(df['misp_ptf']):
                pos='short_btc'
                prix_eth.append(-df['eth_close'].iloc[i])
                prix_btc.append(df['btc_close'].iloc[i])
                nb_eth = (1000/df['eth_close'].iloc[i])
                nb_btc = (1000/df['btc_close'].iloc[i])
                ptf=0
            else:
                pass
        elif pos=='short_eth':
            if df['misp_ptf'].iloc[i]>0 and df['misp_ptf'].iloc[i-1]<=0:
                pos='nul'
                #long eth qu'on a short pr 1000e, short btc qu'on a long pour 1000e
                prix_eth.append(-df['eth_close'].iloc[i])
                prix_btc.append(df['btc_close'].iloc[i])
                benef_eth.append((999.6-nb_eth*df['eth_close'].iloc[i]))
                benef_btc.append((nb_btc*df['btc_close'].iloc[i]-1000.4))
                benef.append((999.6-nb_eth*df['eth_close'].iloc[i]) + (nb_btc*df['btc_close'].iloc[i]-1000.4))
                ptf=2000
        else:
            if df['misp_ptf'].iloc[i]<0 and df['misp_ptf'].iloc[i-1]>=0:
                pos='nul'
                #short eth qu'on a long pr 1000e, long btc qu'on a short pour 1000e
                prix_eth.append(df['eth_close'].iloc[i])
                prix_btc.append(-df['btc_close'].iloc[i])
                benef_eth.append((nb_eth*df['eth_close'].iloc[i]-1000.4))
                benef_btc.append((999.6-nb_btc*df['btc_close'].iloc[i]))
                benef.append((nb_eth*df['eth_close'].iloc[i]-1000.4) + (999.6-nb_btc*df['btc_close'].iloc[i]))
                ptf=2000
    print(f'benef : {benef}')
    
    #else:
        #print('NO cointegration')
    return df, benef, benef_eth, benef_btc
        
        