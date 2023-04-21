from flask import Flask, render_template, request
from bot import test_cointeg
from backtest import backtest
import matplotlib
from broker import get_closed_orders, api
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import numpy as np
import alpaca_trade_api as tradeapi

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():

    # If method not post so display HTML template
    return render_template('index.html')

@app.route('/profil', methods=['GET', 'POST'])
def profil():
    
    # If method not post so display HTML template
    return render_template('profil.html')

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
    df, bnf, et, bt = backtest(bckticket1, bckticket2, bckinterval, bcksize)
    sum_bnf=np.cumsum(bnf)
    print(sum_bnf)
    plt.plot(sum_bnf)
    plt.savefig('./static/backtest.png')
    benefmax = np.sum(bnf)
    print(f"et = {et}")
    print(f"bt = {bt}")
    print(f"benefmax = {benefmax}")
    
    return render_template('display_backtest_result.html', bnf=bnf, et=et, bt=bt, benefmax=benefmax)



# Define route to display the closed orders
@app.route('/dashboard')
def dashboard():
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
    # Render the HTML template with the orders data
    return render_template('dashboard.html', orders=orders_data,positions=positions_data)