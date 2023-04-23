import alpaca_trade_api as tradeapi
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

api = tradeapi.REST('PKMPY03941SVWYJA1MHF', 'h1v7bSV7p5HSaRruYRJFZSUBJ8IQg3rgweqbNmQC', 'https://paper-api.alpaca.markets')
trading_client = TradingClient('PKMPY03941SVWYJA1MHF', 'h1v7bSV7p5HSaRruYRJFZSUBJ8IQg3rgweqbNmQC', paper=True)

def get_closed_orders(api):
    # Get the last 100 closed orders
    closed_orders = api.list_orders(
        status='closed',
        limit=100,
        nested=True  # show nested multi-leg orders
    )

    # Extract the relevant information from each order
    orders_data = []
    for order in closed_orders:
        data = {
            'symbol': order.symbol,
            'qty': order.qty,
            'side': order.side,
            'filled_avg_price': order.filled_avg_price,
            'filled_qty': order.filled_qty,
            'status': order.status,
            'created_at': order.created_at,
            'filled_at': order.filled_at
        }
        orders_data.append(data)

    return orders_data

#BUY order
def buy_order():
    # preparing market order
    market_order_data = MarketOrderRequest(
                    symbol="SPY",
                    qty=0.023,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                    )

    # Market order
    market_order = trading_client.submit_order(
                    order_data=market_order_data
                )

    # Get our order using its Client Order ID.
    my_order = api.get_order_by_client_order_id('my_first_buy')
    print('Got buy #{}'.format(my_order.id))

#SHORT order
def short_order():
    # preparing orders
    market_order_data = MarketOrderRequest(
                        symbol="SPY",
                        qty=1,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.GTC
                        )

    # Market order
    market_order = trading_client.submit_order(
                    order_data=market_order_data
                )

    # Get our order using its Client Order ID.
    my_order = api.get_order_by_client_order_id('my_first_short')
    print('Got short #{}'.format(my_order.id))
