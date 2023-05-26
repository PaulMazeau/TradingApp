from flask import Flask, render_template, request
from bot import test_cointeg
from backtest import Backtest
import matplotlib
from broker import get_closed_orders, api, account
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import numpy as np
import alpaca_trade_api as tradeapi
import plotly.graph_objs as go
import plotly.express as px
from exportData import exportData
from alpaca_trade_api.rest import REST, TimeFrame

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    fond_total = account.equity
    fond_dispo = account.buying_power
    PnL = float(account.cash) - float(account.equity)

    orders_data = get_closed_orders(api)

    # Get the portfolio
    portfolio = api.list_positions()

    # Extract the relevant information from each position
    positions_data = []
    for position in portfolio:
        data = {
            'symbol': position.symbol,
            'qty': position.qty,
            'market_value': position.market_value,
            'average_entry_price': position.avg_entry_price,
            'unrealized_pl': position.unrealized_pl,
            'unrealized_plpc': position.unrealized_plpc,
            'current_price': position.current_price,
            'lastday_price': position.lastday_price,
            'change_today': position.change_today,
        }
        positions_data.append(data)


    # Créer un dictionnaire pour compter le nombre de positions pour chaque symbole
    symbols_counts = {}
    for position in portfolio:
        symbol = position.symbol
        if symbol in symbols_counts:
            symbols_counts[symbol] += 1
        else:
            symbols_counts[symbol] = 1
    # If method not post so display HTML template
    return render_template('index.html', orders=orders_data,positions=positions_data, fond_dispo=fond_dispo, fond_total=fond_total, PnL=PnL)

@app.route('/profil', methods=['GET', 'POST'])
def profil():
    
    # If method not post so display HTML template
    return render_template('profil.html')

@app.route('/ordre')
def ordre():
    
    # If method not post so display HTML template
    return render_template('ordre.html')

@app.route('/backtest', methods=['GET', 'POST'])
def set_backtest():
    return render_template('backtest.html')

@app.route('/bot', methods=['GET', 'POST'])
def bot():
    return render_template('bot.html')

@app.route('/result', methods=['POST'])
def result():
    # Get the form data
    ticket1 = request.form['ticket1']
    ticket2 = request.form['ticket2']
    interval = request.form['interval']
    size = int(request.form['size'])

    # Run the test_cointeg function with the form data
    df, coint = test_cointeg(ticket1, ticket2, interval, size)

    # Plot the results
    mispricing_ptf = df['btc_cr_close'] - df['eth_cr_close']
    plt.plot(mispricing_ptf)
    plt.plot(mispricing_ptf)
    plt.plot([np.mean(mispricing_ptf) for _ in range(len(mispricing_ptf))])
    plt.plot([np.mean(mispricing_ptf)+1*np.std(mispricing_ptf) for _ in range(len(mispricing_ptf))])
    plt.plot([np.mean(mispricing_ptf)-1*np.std(mispricing_ptf) for _ in range(len(mispricing_ptf))])
    plt.plot(mispricing_ptf)
    plt.savefig('./static/graphique.png')
    moy = np.mean(mispricing_ptf)
    print(f"moy = {moy}")
    print(f"coint = {coint}")

    # Return the results
    return render_template('result.html', data=df.to_html(), coint=coint, moy=moy)



@app.route('/display_backtest_result', methods=['POST'])
def display_backtest_result():
    bckticket1 = request.form['bckticket1']
    bckticket2 = request.form['bckticket2']
    bckinterval = request.form['bckinterval']
    bcksize = int(request.form['bcksize'])

    #return the backtest results   
    df, bnf, et, bt = Backtest(bckticket1, bckticket2, bckinterval, bcksize)
    sum_bnf=np.cumsum(bnf)
    print(sum_bnf)
    plt.plot(sum_bnf)
    plt.savefig('./static/backtest.png')
    benefmax = np.sum(bnf)
    print(f"et = {et}")
    print(f"bt = {bt}")
    print(f"benefmax = {benefmax}")
    
    return render_template('display_backtest_result.html', bnf=bnf, et=et, bt=bt, benefmax=benefmax)



@app.route('/data', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        tck1 = request.form.get('tck1')
        startDate = request.form.get('startDate')
        endDate = request.form.get('endDate')

        exportData(tck1, startDate, endDate)

    # Render the HTML template with the orders data
    return render_template('data.html',)

