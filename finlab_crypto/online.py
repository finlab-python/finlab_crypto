"""Live trading on Binance spot: strategy signals -> target positions -> orders.

Market data (prices, symbols, klines) comes from Binance's public endpoint and
works anywhere. Reading balances and placing orders needs an API key and
``python-binance`` (``pip install "finlab_crypto[trade]"``), and must run
where Binance serves its API: api.binance.com refuses US IP addresses
(HTTP 451), which includes Google Colab and most US cloud regions.

Without an API key, ``TradingPortfolio`` runs as a dry run on an empty
account: signals, positions and orders are computed, nothing is sent.
"""
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from finlab_crypto.crawler import get_all_binance, get_exchange_info, get_nbars_binance, get_symbol_tickers

ORDER_MODES = ('TEST', 'MARKET', 'LIMIT')
DEFAULT_STABLE_COIN = 'USDT'
DEFAULT_REBALANCE_THRESHOLD = 0.03
DEFAULT_MIN_NOTIONAL = 10.0
STEP_SIZE_DECIMALS = 9
AMOUNT_TOLERANCE = 1e-4
TRANSACTION_TOLERANCE = 1e-3
SIMPLE_EARN_PREFIX = 'LD'
RESTRICTED_LOCATION_STATUS = 451


def _binance_client(key, secret):
    try:
        from binance.client import Client
        from binance.exceptions import BinanceAPIException
    except ImportError as error:
        raise ImportError('Trading needs python-binance: pip install "finlab_crypto[trade]"') from error
    try:
        return Client(api_key=key, api_secret=secret)
    except BinanceAPIException as error:
        if error.status_code == RESTRICTED_LOCATION_STATUS:
            raise RuntimeError('api.binance.com refuses this location (HTTP 451, e.g. US IPs such as Colab). '
                               'Place orders from a machine outside the US, e.g. your own computer or a '
                               'Cloud Function in asia-east1.') from error
        raise


class TickerInfo:
    """Symbols, prices and (optionally) account balances.

    Args:
      client: python-binance ``Client`` with API key, or None for an empty account.
    """

    def __init__(self, client=None):
        self.exinfo = get_exchange_info()
        self.tickers = get_symbol_tickers()
        self.info = client.get_account() if client else {'balances': []}
        self._symbols = {s['symbol']: s for s in self.exinfo['symbols']}
        self._prices = {t['symbol']: float(t['price']) for t in self.tickers}

    def get_base_asset(self, symbol):
        """Base asset of a symbol, ex: 'ETH' for 'ETHBTC'."""
        return self._symbols[symbol]['baseAsset']

    def get_quote_asset(self, symbol):
        """Quote asset of a symbol, ex: 'BTC' for 'ETHBTC'."""
        return self._symbols[symbol]['quoteAsset']

    def has_symbol(self, symbol):
        return symbol in self._prices

    def get_price(self, symbol):
        return self._prices[symbol]

    def get_asset_price_in_btc(self, asset):
        """Price of one unit of ``asset`` in BTC, or None without a BTC market."""
        if asset == 'BTC':
            return 1
        if self.has_symbol(asset + 'BTC'):
            return self.get_price(asset + 'BTC')
        if self.has_symbol('BTC' + asset):
            return 1 / self.get_price('BTC' + asset)
        return None

    def get_filters(self, symbol):
        """Lot size, step size and minimum order value of a symbol."""
        filters = {f['filterType']: f for f in self._symbols[symbol]['filters']}
        notional = filters.get('MIN_NOTIONAL') or filters.get('NOTIONAL') or {'minNotional': DEFAULT_MIN_NOTIONAL}
        return {
            'min_lot_size': float(filters['LOT_SIZE']['minQty']),
            'step_size': float(filters['LOT_SIZE']['stepSize']),
            'min_notional': float(notional['minNotional']),
        }

    def balances(self):
        """Free spot balances, excluding Simple Earn (``LD*``) positions."""
        free = {b['asset']: float(b['free']) for b in self.info['balances']
                if float(b['free']) != 0 and not b['asset'].startswith(SIMPLE_EARN_PREFIX)}
        return pd.Series(free, index=pd.Index(list(free), dtype=object), dtype=float)


class TradingMethod:
    """A strategy applied to a list of symbols with a position size.

    Args:
      symbols: Trading pairs, ex: ['ETHBTC', 'ADABTC'].
      freq: Candle interval, ex: '4h'.
      lookback: Number of candles used to compute the latest signal.
      strategy: A ``@Strategy``.
      variables: Strategy parameters, ex: {'name': 'sma', 'n1': 30, 'n2': 130}.
      weight_btc: Position size per symbol in BTC, or {'SYMBOL': size, 'default': size}.
      weight, weight_unit: Position size in another unit, ex: weight=500, weight_unit='USDT'.
      filters: Filters passed to ``strategy.backtest``.
      name: Display name.
      execution_price: 'close' or 'open', see ``Strategy.backtest``.
    """

    def __init__(self, symbols, freq, lookback, strategy, variables, weight_btc=None, weight=None,
                 weight_unit=None, filters=None, name='', execution_price='close'):
        if weight_btc is None and weight is None:
            raise ValueError('weight_btc or weight is missing')
        if weight_btc is not None and weight is not None:
            raise ValueError('weight_btc and weight should not be assigned at the same time')

        self.symbols = symbols
        self.freq = freq
        self.lookback = lookback
        self.strategy = strategy
        self.variables = variables
        self.weight_btc = weight_btc
        self.weight = weight_btc if weight_btc is not None else weight
        self.weight_unit = 'BTC' if weight_btc is not None else weight_unit
        self.filters = filters
        self.name = name
        self.execution_price = execution_price

    def weight_of(self, symbol):
        """Position size of ``symbol`` in ``weight_unit``."""
        if isinstance(self.weight, dict):
            return self.weight.get(symbol, self.weight['default'])
        return self.weight


class TradingPortfolio:
    """Several TradingMethods sharing one Binance account.

    Args:
      binance_key, binance_secret: API key pair. Leave empty for a dry run.
      execute_before_candle_complete: Use the still-forming candle for signals.
    """

    def __init__(self, binance_key=None, binance_secret=None, execute_before_candle_complete=False):
        self._client = _binance_client(binance_key, binance_secret) if binance_key else None
        self._trading_methods = []
        self._margins = {}
        self.ticker_info = TickerInfo(self._client)
        self.default_stable_coin = DEFAULT_STABLE_COIN
        self.execute_before_candle_complete = execute_before_candle_complete

    @property
    def is_dry_run(self):
        return self._client is None

    def set_default_stable_coin(self, token):
        self.default_stable_coin = token

    def register(self, trading_method):
        """Add a TradingMethod."""
        if trading_method.execution_price == 'open' and self.execute_before_candle_complete:
            raise ValueError('execution_price="open" needs complete candles: '
                             'set execute_before_candle_complete=False and trade right after candles close')
        self._trading_methods.append(trading_method)

    def register_margin(self, asset, weight_btc):
        """Keep an extra ``weight_btc`` amount of ``asset`` (ex: 'USDT', 1000) outside the strategies."""
        self._margins[asset] = weight_btc

    def _results(self, ohlcvs):
        """Backtest every (method, symbol) pair on ``ohlcvs``."""
        rows = []
        for method in self._trading_methods:
            for symbol in method.symbols:
                portfolio = method.strategy.backtest(ohlcvs[(symbol, method.freq)], method.variables,
                                                     filters=method.filters, freq=method.freq)
                rows.append({
                    'name': method.name,
                    'symbol': symbol,
                    'freq': method.freq,
                    'weight': method.weight_of(symbol),
                    'portfolio': portfolio,
                    'trading_method': method,
                    'signal': portfolio.cash().iloc[-1] == 0,
                })
        return pd.DataFrame(rows)

    def get_all_symbol_lookback(self):
        """{(symbol, freq): lookback} of every price series the methods need."""
        lookbacks = {}
        conversions = {}
        max_lookback = max((m.lookback for m in self._trading_methods), default=0)
        for method in self._trading_methods:
            for symbol in method.symbols:
                key = (symbol, method.freq)
                lookbacks[key] = max(lookbacks.get(key, 0), method.lookback)
                base_asset = self.ticker_info.get_base_asset(symbol)
                if base_asset != method.weight_unit:
                    key = (base_asset + method.weight_unit, method.freq)
                    conversions[key] = max(conversions.get(key, 0), method.lookback)
            if method.weight_unit != 'BTC':
                conversions[('BTC' + method.weight_unit, method.freq)] = max_lookback
        return {**lookbacks, **conversions}

    def get_ohlcvs(self):
        """Latest ``lookback`` candles of every needed symbol."""
        return {(symbol, freq): get_nbars_binance(symbol, freq, lookback)
                for (symbol, freq), lookback in self.get_all_symbol_lookback().items()}

    def get_full_ohlcvs(self):
        """Full history of every needed symbol (cached in ./history/)."""
        return {(symbol, freq): get_all_binance(symbol, freq) for symbol, freq in self.get_all_symbol_lookback()}

    @staticmethod
    def _price(ohlcvs, symbol, freq, column, time=None):
        """Price of ``symbol`` at ``time`` (latest if None); 1 when no conversion is needed."""
        if symbol is None:
            return 1
        prices = ohlcvs[(symbol, freq)][column]
        return prices.iloc[-1] if time is None else prices.loc[time]

    def get_latest_signals(self, ohlcvs, html=False):
        """Latest signal, entry and value (in BTC) of every (method, symbol).

        Args:
          ohlcvs: Output of ``get_ohlcvs()``.
          html: Also save each strategy chart to ``{symbol}-{freq}-{name}.html``.
        """
        rows = []
        now = pd.Timestamp.now(tz='UTC')
        for method in self._trading_methods:
            price_column = method.execution_price
            price_in_unit = partial(self._price, ohlcvs, freq=method.freq, column=price_column)
            for symbol in method.symbols:
                ohlcv = ohlcvs[(symbol, method.freq)]
                if not self.execute_before_candle_complete and price_column == 'close':
                    ohlcv = ohlcv.loc[:now - (ohlcv.index[-1] - ohlcv.index[-2])]

                html_name = f'{symbol}-{method.freq}-{method.name}.html' if html else None
                result = method.strategy.backtest(ohlcv, method.variables, filters=method.filters, plot=html,
                                                  html=html_name, freq=method.freq, fees=0., slippage=0.,
                                                  execution_price=price_column)
                signal = result.cash().iloc[-1] == 0
                weight = method.weight_of(symbol)

                base_asset = self.ticker_info.get_base_asset(symbol)
                base_symbol = None if base_asset == method.weight_unit else base_asset + method.weight_unit
                btc_symbol = None if method.weight_unit == 'BTC' else 'BTC' + method.weight_unit

                entry_price = return_ = amount = value_in_btc = 0
                entry_time = 0
                if signal:
                    orders = result.orders.records
                    entry_price = orders['price'].iloc[-1]
                    execution_offset = 1 if price_column == 'open' else 0
                    entry_time = ohlcv.index[int(orders['idx'].iloc[-1]) + execution_offset]
                    return_ = ohlcv[price_column].iloc[-1] / entry_price - 1

                    base_then, base_now = price_in_unit(base_symbol, time=entry_time), price_in_unit(base_symbol)
                    btc_then, btc_now = price_in_unit(btc_symbol, time=entry_time), price_in_unit(btc_symbol)
                    weight_btc = weight / btc_then
                    value_in_btc = weight_btc / base_then * base_now / (btc_now / btc_then)
                    amount = weight_btc / (base_then / btc_then)
                    present_amount = value_in_btc / (base_now / btc_now)
                    assert abs(amount / present_amount - 1) < AMOUNT_TOLERANCE
                else:
                    weight_btc = weight / price_in_unit(btc_symbol)

                rows.append({
                    'symbol': symbol,
                    'method name': method.name,
                    'latest_signal': signal,
                    'weight_btc': weight_btc,
                    'freq': method.freq,
                    'return': return_,
                    'amount': amount,
                    'value_in_btc': value_in_btc,
                    'latest_price': ohlcv[price_column].iloc[-1],
                    'entry_price': entry_price,
                    'entry_time': entry_time,
                    'html': html_name,
                })
        return pd.DataFrame(rows)

    def calculate_position_size(self, signals, rebalance_threshold=DEFAULT_REBALANCE_THRESHOLD, excluded_assets=()):
        """Compare target positions with the account and derive orders.

        Args:
          signals: Output of ``get_latest_signals()``.
          rebalance_threshold: Skip adjustments smaller than this fraction of the target.
          excluded_assets: Assets never traded (the default stable coin always is).

        Returns:
          (position, position_in_btc, orders): target vs. present position per
          asset (in units and in BTC), and the orders to place.
        """
        info = self.ticker_info
        excluded_assets = {*excluded_assets, self.default_stable_coin}

        signals = signals.copy()
        signals['base_asset'] = signals.symbol.map(info.get_base_asset)
        signals['quote_asset'] = signals.symbol.map(info.get_quote_asset)
        signals['base_value_btc'] = signals.latest_signal * signals.value_in_btc
        signals['quote_value_btc'] = -(signals.latest_signal.astype(int) * signals.weight_btc)
        quote_assets = set(signals.quote_asset)

        base_value = signals.groupby('base_asset').base_value_btc.sum()
        quote_value = signals.groupby('quote_asset').quote_value_btc.sum()
        position = info.balances()
        all_assets = base_value.index.union(quote_value.index).union(position.index)

        algo_value_btc = base_value.reindex(all_assets, fill_value=0) + quote_value.reindex(all_assets, fill_value=0)
        position = position.reindex(all_assets, fill_value=0)
        price_in_btc = pd.Series(all_assets.map(info.get_asset_price_in_btc), index=all_assets, dtype=float)
        margin_btc = pd.Series(self._margins, dtype=float).reindex(all_assets, fill_value=0) * price_in_btc
        estimate_btc = algo_value_btc + margin_btc

        position_btc = pd.DataFrame({
            'algo_p': algo_value_btc,
            'margin_p': margin_btc,
            'estimate_p': estimate_btc,
            'present_p': position * price_in_btc,
            'difference': estimate_btc.clip(0, None) - position * price_in_btc,
            'rebalance_threshold': estimate_btc.abs() * rebalance_threshold,
        })
        position_btc['rebalance'] = position_btc.difference.abs() > position_btc.rebalance_threshold
        traded = all_assets.isin(signals.quote_asset) | all_assets.isin(signals.base_asset)
        position_btc['excluded'] = ~traded | all_assets.isin(list(excluded_assets))

        rebalance_btc = position_btc.difference.where(
            (position_btc.rebalance | all_assets.isin(list(quote_assets))) & ~position_btc.excluded, 0)
        increase = rebalance_btc[rebalance_btc > 0]
        decrease = rebalance_btc[rebalance_btc < 0]

        pair_orders_btc = {}
        for buy_asset, buy_value in increase.items():
            for sell_asset, sell_value in decrease.items():
                value = min(-sell_value, buy_value)
                if info.has_symbol(sell_asset + buy_asset) and buy_asset in quote_assets:
                    symbol, signed_value = sell_asset + buy_asset, -value
                elif info.has_symbol(buy_asset + sell_asset) and sell_asset in quote_assets:
                    symbol, signed_value = buy_asset + sell_asset, value
                else:
                    continue
                increase.loc[buy_asset] -= value
                decrease.loc[sell_asset] += value
                pair_orders_btc[symbol] = signed_value

        stable_orders_btc = pd.concat([increase, decrease])
        stable_orders_btc.index = stable_orders_btc.index + self.default_stable_coin
        stable_orders_btc = stable_orders_btc.drop(self.default_stable_coin * 2, errors='ignore')
        orders = pd.concat([stable_orders_btc, pd.Series(pair_orders_btc, dtype=float)]).to_frame('value_in_btc')

        orders['base_asset'] = orders.index.map(info.get_base_asset)
        orders['quote_asset'] = orders.index.map(info.get_quote_asset)
        orders['value'] = orders.value_in_btc / orders.base_asset.map(info.get_asset_price_in_btc)
        orders['price'] = orders.index.map(info.get_price)
        orders = orders.groupby(level=0).agg(
            {'value_in_btc': 'sum', 'value': 'sum', 'base_asset': 'first', 'quote_asset': 'first', 'price': 'first'})
        orders = orders[orders.value != 0]
        self._verify_orders(orders, position_btc, all_assets)

        is_quote_asset_order = orders.base_asset.isin(list(quote_assets))
        below_threshold = orders.value_in_btc.abs() < position_btc.rebalance_threshold.reindex(orders.base_asset).values
        orders = orders[~(is_quote_asset_order & below_threshold)]

        position = position_btc.copy()
        position[position.columns[:-2]] = position_btc.iloc[:, :-2].div(price_in_btc, axis=0)
        return position, position_btc, self._apply_exchange_filters(orders)

    def _verify_orders(self, orders, position_btc, all_assets):
        """Orders must move every asset by exactly its difference."""
        moved = (orders.value_in_btc.groupby(orders.base_asset).sum().reindex(all_assets, fill_value=0)
                 - orders.value_in_btc.groupby(orders.quote_asset).sum().reindex(all_assets, fill_value=0))
        moved = moved[moved != 0].drop(self.default_stable_coin, errors='ignore')
        mismatch = (moved / position_btc.difference.reindex(moved.index) - 1).abs() >= TRANSACTION_TOLERANCE
        if mismatch.any():
            raise RuntimeError(f'Orders do not match the target position:\n{orders}\n{position_btc}')

    def _apply_exchange_filters(self, orders):
        """Round to the step size and drop orders below Binance's minimums."""
        if orders.empty:
            return orders.assign(final_value=pd.Series(dtype=float), final_value_in_btc=pd.Series(dtype=float))

        filters = pd.DataFrame({symbol: self.ticker_info.get_filters(symbol) for symbol in orders.index}).T
        quantity = ((orders.value / filters.step_size).astype(int) * filters.step_size).round(STEP_SIZE_DECIMALS)
        quantity[quantity.abs() < filters.min_lot_size] = 0
        quantity[quantity.abs() * orders.price < filters.min_notional] = 0

        orders = orders.assign(
            final_value=quantity,
            final_value_in_btc=quantity * orders.base_asset.map(self.ticker_info.get_asset_price_in_btc))
        return orders[orders.final_value != 0]

    def execute_orders(self, transactions, mode='TEST'):
        """Send orders to Binance.

        Args:
          transactions: ``orders`` from ``calculate_position_size()``.
          mode: 'TEST' validates orders without trading (only prints them in
            a dry run), 'MARKET' trades at market price, 'LIMIT' places limit
            orders at the latest price.

        Returns:
          pd.DataFrame: One row per order with its result.
        """
        if mode not in ORDER_MODES:
            raise ValueError(f'mode should be one of {ORDER_MODES}')
        if self.is_dry_run and mode != 'TEST':
            raise RuntimeError(f'{mode} orders need a Binance API key; without one only mode="TEST" is available')

        print('|---------EXECUTION LOG----------|')
        print('| time:', pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
              '(dry run, nothing sent)' if self.is_dry_run else '')

        trades = {}
        for symbol, lot in transactions.final_value.items():
            order = {'side': 'BUY' if lot > 0 else 'SELL', 'type': 'MARKET', 'symbol': symbol,
                     'quantity': abs(lot)}
            if mode == 'LIMIT':
                order.update(type='LIMIT', price=transactions.price.loc[symbol], timeInForce='GTC')
            trades[symbol] = {**order, 'result': self._send(order, mode)}
            print('|', mode, symbol, order['side'], order['quantity'], trades[symbol]['result'])
        return pd.DataFrame(trades).T

    def _send(self, order, mode):
        if self.is_dry_run:
            return 'dry run'
        for open_order in self._client.get_open_orders(symbol=order['symbol']):
            self._client.cancel_order(symbol=order['symbol'], orderId=open_order['orderId'])
        send = self._client.create_test_order if mode == 'TEST' else self._client.create_order
        try:
            send(**order)
        except Exception as error:  # report every rejected order instead of stopping the batch
            return f'FAIL: {error}'
        return 'success'

    def status(self, ohlcvs):
        """Show every strategy's equity and signal; returns a widget to inspect one of them."""
        import ipywidgets as widgets

        results = self._results(ohlcvs)
        equity = pd.DataFrame({f'{r.name}-{r.symbol}-{r.freq}': r.portfolio.cumulative_returns()
                               for r in results.itertuples()})
        (equity + 1).plot(logy=True, figsize=(15, 5), title='Cumulative returns')
        plt.show()
        display(results.drop(columns=['portfolio', 'trading_method']))

        method_dropdown = widgets.Dropdown(options=[(m.name, m) for m in self._trading_methods])
        symbol_dropdown = widgets.Dropdown(options=[(f'{s}-{f}', (s, f)) for s, f in ohlcvs])
        backtest_button = widgets.Button(description='backtest')
        backtest_panel = widgets.Output()

        @backtest_panel.capture(clear_output=True)
        def backtest(_):
            method = method_dropdown.value
            symbol, freq = symbol_dropdown.value
            method.strategy.backtest(ohlcvs[(symbol, freq)], variables=method.variables, filters=method.filters,
                                     freq=freq, plot=True)

        backtest_button.on_click(backtest)
        return widgets.VBox([widgets.HBox([method_dropdown, symbol_dropdown, backtest_button]), backtest_panel])

    def portfolio_backtest(self, ohlcvs, min_freq, fee=0.002, delay=0):
        """Backtest the whole account in USDT, with every method's position size.

        Args:
          ohlcvs: Output of ``get_full_ohlcvs()``.
          min_freq: Rebalance frequency, ex: '4h'.
          fee: Cost per unit of turnover.
          delay: Delay signals by this many bars.

        Returns:
          pd.DataFrame: The backtest of every (method, symbol).
        """
        info = self.ticker_info
        results = self._results(ohlcvs)

        holdings = {}
        for r in results.itertuples():
            weight_btc = r.weight * info.get_asset_price_in_btc(r.trading_method.weight_unit)
            in_market = (r.portfolio.cash() == 0).shift(delay, fill_value=False).astype(float)
            holdings[(r.symbol, r.Index)] = in_market * weight_btc
        holdings = pd.DataFrame(holdings).resample(min_freq).last().ffill().fillna(0)
        symbols = holdings.columns.get_level_values(0)

        def by_asset(frame, assets):
            return frame.T.groupby(list(assets)).sum().T

        base_position = by_asset(holdings, symbols.map(info.get_base_asset))
        quote_position = -by_asset(holdings, symbols.map(info.get_quote_asset))
        assets = base_position.columns.union(quote_position.columns).union(list(self._margins)).union(['USDT'])
        position = base_position.reindex(columns=assets, fill_value=0) + quote_position.reindex(columns=assets,
                                                                                                 fill_value=0)

        initial_margin_btc = 0
        for asset, amount in self._margins.items():
            position[asset] += info.get_asset_price_in_btc(asset) * amount
            initial_margin_btc += info.get_asset_price_in_btc(asset) * amount

        shortfall = position.drop(columns='USDT').clip(None, 0).sum(axis=1)
        usdt = position.USDT + shortfall
        position = position.clip(0, None)
        position['USDT'] = usdt
        usdt_in_btc = info.get_asset_price_in_btc('USDT')
        extra_usdt = -min(position.USDT.min(), 0) / usdt_in_btc
        if extra_usdt > 0:
            print('WARNING: additional USDT is required:', extra_usdt, 'USD')

        changes = position.index[(position != position.shift()).any(axis=1)].union(position.index[-1:])
        position = position.loc[changes]
        position.index = position.index.tz_localize(None)

        close = pd.DataFrame({asset: get_all_binance(asset + 'USDT', min_freq).close
                              for asset in position.columns if asset != 'USDT'})
        close.index = close.index.tz_localize(None)
        rebalance_times = position.index.intersection(close.index)
        close = close.loc[rebalance_times]
        position = position.loc[rebalance_times]

        asset_return = (close.pct_change().shift(-1).fillna(0) * position
                        - fee * (position - position.shift()).abs()).fillna(0)

        (asset_return.cumsum() / usdt_in_btc).plot(figsize=(15, 4), title='Profit by asset (USDT)')
        plt.show()
        equity = (asset_return.sum(axis=1).cumsum() + initial_margin_btc) / usdt_in_btc
        equity.plot(figsize=(15, 4), title='Account value (USDT)')
        plt.show()
        (equity / equity.cummax() - 1).plot(figsize=(15, 3), title='Drawdown')
        plt.show()
        return results


def render_html(signals, position, position_btc, orders, order_results):
    """HTML report of one rebalance, used as the Cloud Function response."""
    tables = [('Strategy signals', signals), ('Position', position), ('Position in BTC', position_btc)]
    body = ''.join(f'<h2>{title}</h2>{table.to_html(classes="pure-table pure-table-horizontal")}'
                   for title, table in tables)
    if len(orders):
        orders_table = orders.assign(result=order_results['result'])
        body += f'<h2>Orders</h2>{orders_table.to_html(classes="pure-table pure-table-horizontal")}'
    else:
        body += '<h2>Orders</h2><p>None</p>'

    return f"""<!DOCTYPE html>
<html>
<head>
  <title>Crypto Portfolio</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/purecss@3.0.0/build/pure-min.css">
</head>
<body style="padding: 5vw">
  <h1>Crypto Portfolio</h1>
  {body}
  <br>
  <button onclick="updatePosition('MARKET')">place market orders</button>
  <button onclick="updatePosition('LIMIT')">place limit orders</button>
  <script>
    function updatePosition(mode) {{
      window.location = window.location.href.split('?')[0] + '?mode=' + mode;
    }}
  </script>
</body>
</html>"""
