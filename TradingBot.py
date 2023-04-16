from flask import Flask, render_template, request
from bot import test_cointeg
from backtest import backtest
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import numpy as np

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():

    # If method not post so display HTML template
    return render_template('index.html')

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


@app.route('/backtest_route', methods=['POST'])
def backtest_route():
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
    
    return render_template('backtest.html', bnf=bnf, et=et, bt=bt, benefmax=benefmax)
