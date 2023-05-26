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



# Define route to display the closed orders
@app.route('/data')
def data():
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

    # Créer une liste de labels et de valeurs pour le graphique camembert du portfolio
    labels_portfolio = list(symbols_counts.keys())
    values_portfolio = list(symbols_counts.values())

    # Créer le graphique camembert du portfolio avec Plotly
    fig_portfolio = go.Figure(data=[go.Pie(labels=labels_portfolio, values=values_portfolio)])

    # Configurer le layout du graphique du portfolio
    fig_portfolio.update_layout(
        title='Répartition du portfolio par symbole',
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    fig_portfolio.update_traces(marker=dict(colors=['#7354F3', '#B1496A', '#0000FF']))


    # Convertir le graphique du portfolio en HTML
    graph_html_portfolio = fig_portfolio.to_html(full_html=False)

    # Render the HTML template with the orders data
    return render_template('data.html', orders=orders_data,positions=positions_data, graph_html_portfolio=graph_html_portfolio)