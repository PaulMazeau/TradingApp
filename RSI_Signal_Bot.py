import yfinance as yf
import ta
import threading
import requests
import os
from dotenv import load_dotenv

load_dotenv()  # Assurez-vous d'appeler cette fonction pour charger les variables d'environnement

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
lock = threading.Lock()

def send_telegram_message(chat_id, token, message):
    base_url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={message}"
    requests.get(base_url)

def get_rsi(ticker_symbol, interval='1d', period=14):
    with lock:
        stock_data = yf.download(ticker_symbol, interval=interval)
    rsi = ta.momentum.RSIIndicator(stock_data['Close'], window=period).rsi()
    return round(rsi.iloc[-1], 1)

def monitor_rsi(ticker_symbols, threshold_low=30, threshold_high=60):
    for ticker_symbol in ticker_symbols:
        rsi = get_rsi(ticker_symbol)
        if rsi < threshold_low:
            message = f'{ticker_symbol}: Achat, RSI = {rsi}'
            print(message)
            send_telegram_message(CHAT_ID, TELEGRAM_TOKEN, message)
        elif rsi > threshold_high:
            message = f'{ticker_symbol}: Vente, RSI = {rsi}'
            print(message)
            send_telegram_message(CHAT_ID, TELEGRAM_TOKEN, message)
        else:
            print(f'{ticker_symbol}: Attente... RSI actuel: {rsi}')

if __name__ == '__main__':
    ticker_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']  # Ajoutez autant de symboles que vous le souhaitez
    for ticker in ticker_symbols:
        thread = threading.Thread(target=monitor_rsi, args=([ticker],))
        thread.start()
        thread.join()  # Attendez que le thread se termine avant de passer au suivant
