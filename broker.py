import alpaca_trade_api as tradeapi

api = tradeapi.REST('PKMPY03941SVWYJA1MHF', 'h1v7bSV7p5HSaRruYRJFZSUBJ8IQg3rgweqbNmQC', 'https://paper-api.alpaca.markets')

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

  