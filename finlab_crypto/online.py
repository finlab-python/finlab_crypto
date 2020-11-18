import sys
import plotly.express as px
import pandas as pd
import datetime
from IPython.display import display
from binance.enums import *
from finlab_crypto.crawler import get_nbars_binance, get_all_binance
from binance.client import Client

class TickerInfo():

  def __init__(self, client):
    self.exinfo = client.get_exchange_info()
    self.info = client.get_account()
    self.tickers = client.get_symbol_ticker()

  @staticmethod
  def _list_select(list, key, value):
    ret = [l for l in list if l[key] == value]
    if len(ret) == 0:
      return None
    else:
      return ret[0]

  def get_base_asset(self, symbol):
    sinfo = self._list_select(self.exinfo['symbols'], 'symbol', symbol)
    return sinfo['baseAsset']

  def get_quote_asset(self, symbol):
    sinfo = self._list_select(self.exinfo['symbols'], 'symbol', symbol)
    return sinfo['quoteAsset']

  def get_asset_price_in_btc(self, asset):

    if asset == 'BTC':
      return 1

    ret = self._list_select(self.tickers, 'symbol', asset + 'BTC')

    if ret is not None:
      return float(ret['price'])

    ret = self._list_select(self.tickers, 'symbol', 'BTC' + asset)
    if ret is not None:
      return 1/float(ret['price'])

    return None

class TradingMethod():
  def __init__(self, symbols, freq, lookback, strategy, variables, weight_btc, filters=None, name=''):
    self.symbols = symbols
    self.freq = freq
    self.lookback = lookback
    self.strategy = strategy
    self.variables = variables
    self.weight_btc = weight_btc
    self.filters = filters
    self.name = name

class TradingPortfolio():
  def __init__(self, binance_key, binance_secret):
    self._client = Client(api_key=binance_key, api_secret=binance_secret)
    self._trading_methods = []
    self._margins = {}
    self.ticker_info = TickerInfo(self._client)
    self.quote_asset = 'BTC'


  def register(self, trading_method):
    self._trading_methods.append(trading_method)

  def register_margin(self, asset, weight_btc):
    self._margins[asset] = weight_btc

  def get_all_symbol_lookback(self):

    symbol_lookbacks = {}
    for method in self._trading_methods:
      for a in method.symbols:
        if (a, method.freq) not in symbol_lookbacks or method.lookback > symbol_lookbacks[(a, method.freq)]:
          symbol_lookbacks[(a, method.freq)] = method.lookback

    # add quote asset historical data
    addition = {}
    for (symbol, freq), lookback in symbol_lookbacks.items():
      base_asset = self.ticker_info.get_base_asset(symbol)
      if base_asset != self.quote_asset:
        new_symbol = base_asset + self.quote_asset
        addition[(new_symbol, freq)] = lookback


    return {**symbol_lookbacks, **addition}

  def get_ohlcvs(self):

    symbol_lookbacks = self.get_all_symbol_lookback()

    ohlcvs = {}
    for (symbol, freq), lookback in symbol_lookbacks.items():
      ohlcvs[(symbol, freq)] = get_nbars_binance(symbol, freq, lookback, self._client)

    return ohlcvs

  def get_full_ohlcvs(self):

    symbol_lookbacks = self.get_all_symbol_lookback()
    ohlcvs = {}

    for (symbol, freq), lookback in symbol_lookbacks.items():
      ohlcvs[(symbol, freq)] = get_all_binance(symbol, freq)
    return ohlcvs


  def get_latest_signals(self, ohlcvs):

    ret = []
    for method in self._trading_methods:
      for symbol in method.symbols:
        ohlcv = ohlcvs[(symbol, method.freq)]
        result = method.strategy.backtest(ohlcv,
                method.variables, filters=method.filters, freq=method.freq, fees=0, slippage=0)

        signal = result.cash.iloc[-1] == 0
        return_ = 0
        weight_btc = method.weight_btc
        entry_price = 0
        entry_time = 0
        value_in_btc = 0
        if signal:
            return_ = result.positions.records.iloc[-1]['return']
            entry_price = result.positions.records.iloc[-1]['entry_price']
            entry_time = ohlcv.index[int(result.positions.records.iloc[-1]['entry_idx'])]

            base_asset = self.ticker_info.get_base_asset(symbol)
            if base_asset != self.quote_asset:
                quote_asset_symbol = base_asset + self.quote_asset
                quote_asset_price_previous = ohlcvs[(quote_asset_symbol, method.freq)].close.loc[entry_time]
                quote_asset_price_now = ohlcvs[(quote_asset_symbol, method.freq)].close.iloc[-1]
            else:
                quote_asset_price_previous = 1
                quote_asset_price_now = 1

            value_in_btc = weight_btc / quote_asset_price_previous * quote_asset_price_now

        if isinstance(method.weight_btc, dict):
          if symbol in method.weight_btc:
            weight = method.weight_btc[symbol]
          else:
            weight = method.weight_btc['default']
        else:
          weight = method.weight_btc

        ret.append({
          'symbol': symbol,
          'method name': method.name,
          'latest_signal': signal,
          'weight_btc': weight,
          'freq': method.freq,
          'return': return_,
          'value_in_btc': value_in_btc * signal,
          'latest_price': ohlcv.close.iloc[-1],
          'entry_price': entry_price,
          'entry_time': entry_time,
        })

    ret = pd.DataFrame(ret)
    return ret

  def calculate_position_size(self, signals, rebalance_threshold=0.03, excluded_assets=list()):

    if 'USDT' not in excluded_assets:
        excluded_assets.append('USDT')

    signals['base_asset'] = signals.symbol.map(self.ticker_info.get_base_asset)
    signals['quote_asset'] = signals.symbol.map(self.ticker_info.get_quote_asset)
    signals['base_value_btc'] = signals.latest_signal * signals.value_in_btc
    signals['quote_value_btc'] = -(signals.latest_signal.astype(int) * signals.weight_btc)

    quote_asset_list = list(set(signals.quote_asset))

    # calculate base and quote assets (in btc term)
    base_asset_value = pd.Series(signals.base_value_btc.values, index=signals.base_asset)
    quote_asset_value = pd.Series(signals.quote_value_btc.values, index=signals.quote_asset)
    base_asset_value = base_asset_value.groupby(level=0).sum()
    quote_asset_value = quote_asset_value.groupby(level=0).sum()

    # get position
    position = pd.Series({i['asset']:i['free'] for i in self.ticker_info.info['balances']
        if float(i['free']) != 0}).astype(float)
    position = position[position.index.str[:2] != 'LD']

    # refine asset index
    all_assets = base_asset_value.index | quote_asset_value.index | position.index

    base_asset_value = base_asset_value.reindex(all_assets).fillna(0)
    quote_asset_value = quote_asset_value.reindex(all_assets).fillna(0)
    position = position.reindex(all_assets).fillna(0)

    # calculate algo value
    algo_value_in_btc = base_asset_value + quote_asset_value
    asset_price_in_btc = position.index.map(self.ticker_info.get_asset_price_in_btc)
    algo_value = algo_value_in_btc / asset_price_in_btc

    # calculate diffierence
    margin_position = pd.Series(self._margins).reindex(all_assets).fillna(0)

    diff_value_btc = pd.DataFrame({
      'algo_p': algo_value_in_btc,
      'margin_p': margin_position * asset_price_in_btc,
      'estimate_p': algo_value_in_btc + margin_position * asset_price_in_btc,
      'present_p': position * asset_price_in_btc,
      'difference': (algo_value_in_btc + margin_position * asset_price_in_btc).clip(0, None) - position * asset_price_in_btc,
      'rebalance_threshold': (algo_value_in_btc + margin_position * asset_price_in_btc).abs() * rebalance_threshold,
    })
    diff_value_btc['rebalance'] = diff_value_btc['difference'].abs() >  diff_value_btc['rebalance_threshold']
    diff_value_btc.loc[quote_asset_list, 'rebalance'] = True

    # excluding checking of asset positions

    excluded = pd.Series(True, diff_value_btc.index)
    excluded[diff_value_btc.index.isin(signals.quote_asset) | diff_value_btc.index.isin(signals.base_asset)] = False
    excluded[diff_value_btc.index.isin(excluded_assets)] = True

    diff_value_btc['excluded'] = excluded

    diff_value = diff_value_btc.copy()
    diff_value = diff_value.div(asset_price_in_btc, axis=0)
    diff_value.rebalance = diff_value.rebalance != 0
    diff_value.excluded = diff_value.excluded != 0

    # calculate transaction

    rebalance_value_btc = diff_value_btc.rebalance * diff_value_btc.difference * (~diff_value_btc.excluded)
    increase_asset_amount = rebalance_value_btc[rebalance_value_btc > 0]
    decrease_asset_amount = rebalance_value_btc[rebalance_value_btc < 0]

    diff_value_btc['rebalance'] = diff_value_btc['difference'].abs() >  diff_value_btc['rebalance_threshold']
    diff_value['rebalance'] = diff_value_btc.rebalance

    txn_btc = {}

    for nai, ai in increase_asset_amount.items():
      for nad, ad in decrease_asset_amount.items():

        symbol = nad + nai
        amount = min(-ad, ai)

        is_valid = self.ticker_info._list_select(self.ticker_info.tickers, 'symbol', symbol) is not None and nai in quote_asset_list

        if is_valid:
          increase_asset_amount.loc[nai] -= amount
          decrease_asset_amount.loc[nad] += amount
          txn_btc[symbol] = -amount
          break

        symbol = nai + nad
        is_valid = self.ticker_info._list_select(self.ticker_info.tickers, 'symbol', symbol) is not None and nad in quote_asset_list

        if is_valid:
          increase_asset_amount.loc[nai] -= amount
          decrease_asset_amount.loc[nad] += amount
          txn_btc[symbol] = amount
          break

    # assumption: usdt can be a quote asset for all alt-coins
    transaction_btc = increase_asset_amount.append(decrease_asset_amount)
    transaction_btc.index = transaction_btc.index + 'USDT'

    if 'USDTUSDT' in transaction_btc.index:
      transaction_btc.pop('USDTUSDT')

    transaction_btc = transaction_btc.append(pd.Series(txn_btc))

    transaction = transaction_btc.to_frame(name='value_in_btc')
    transaction['base_asset'] = transaction.index.map(self.ticker_info.get_base_asset)
    transaction['quote_asset'] = transaction.index.map(self.ticker_info.get_quote_asset)
    transaction['value'] = transaction['value_in_btc'] / transaction.base_asset.map(self.ticker_info.get_asset_price_in_btc)
    transaction['price'] = transaction.index.map(
            lambda s: self.ticker_info._list_select(self.ticker_info.tickers, 'symbol', s)['price'])
    transaction = transaction.groupby(level=0).agg(dict(value_in_btc='sum', value='sum', base_asset='first', quote_asset='first', price='first'))
    transaction = transaction[transaction.value != 0]

    # check difference after transaction
    def asset_distributed(v):
      asset_increase = v.value_in_btc.groupby(v.base_asset).sum()
      asset_decrease = v.value_in_btc.groupby(v.quote_asset).sum()
      return asset_increase.reindex(all_assets).fillna(0) - asset_decrease.reindex(all_assets).fillna(0)

    verify_assets = asset_distributed(transaction)
    verify_assets = verify_assets[verify_assets != 0]
    verify = (verify_assets / diff_value_btc.difference.reindex(verify_assets.index) - 1).abs() < 0.001

    try:
      assert verify[verify.index != 'USDT'].all()
    except:
      print(diff_value_btc)
      print(transaction)
      print(verify_assets)
      print(verify)
      raise Exception("validation fail")

    # filter out orders where base asset is in quote asset list (ex: btcusdt)
    # assumption: base asset should only be paired by one quote asset
    transaction = transaction[~(transaction.base_asset.isin(quote_asset_list) & (transaction.value_in_btc.abs() < diff_value_btc.loc[transaction['base_asset']].rebalance_threshold.values))]


    # verify diff_value
    def get_filters(exinfo, symbol):
      filters = self.ticker_info._list_select(self.ticker_info.exinfo['symbols'], 'symbol', symbol)['filters']
      min_lot_size = self.ticker_info._list_select(filters, 'filterType', 'LOT_SIZE')['minQty']
      step_size = self.ticker_info._list_select(filters, 'filterType', 'LOT_SIZE')['stepSize']
      min_notional = self.ticker_info._list_select(filters, 'filterType', 'MIN_NOTIONAL')['minNotional']
      return {
          'min_lot_size': min_lot_size,
          'step_size': step_size,
          'min_notional': min_notional,
      }

    filters = pd.DataFrame({s: get_filters(self.ticker_info.exinfo, s) for s in transaction.index}).transpose().astype(float)

    if len(transaction) != 0:
        min_notional = filters.min_notional
        minimum_lot_size = filters.min_lot_size
        step_size = filters.step_size

        # rebalance filter:
        diff = transaction['value']

        # step size filter
        diff = round((diff / step_size).astype(int) * step_size, 9)

        # minimum lot filter
        diff[diff.abs() < minimum_lot_size] = 0

        # minimum notional filter
        diff[diff.abs() * transaction.price.astype(float) < min_notional] = 0

        transaction['final_value'] = diff
        transaction['final_value_in_btc'] = diff * transaction.base_asset.map(self.ticker_info.get_asset_price_in_btc)
    else:
        transaction = pd.DataFrame(None, columns=['final_value'])

    transaction = transaction[transaction['final_value'] != 0]

    return diff_value, diff_value_btc, transaction


  def execute_orders(self, transactions, mode='TEST'):

    def cancel_orders(symbol):
      orders = self._client.get_open_orders(symbol=symbol)
      for o in orders:
        self._client.cancel_order(symbol=symbol, orderId=o['orderId'])

    order_func = self._client.create_order if mode == 'MARKET' or mode == 'LIMIT' else self._client.create_test_order

    print('|---------EXECUTION LOG----------|')
    print('| time: ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    trades = {}
    for s, lot in transactions.final_value.items():

      cancel_orders(s)

      if lot == 0:
        continue

      side = SIDE_BUY if lot > 0 else SIDE_SELL
      try:
        args = dict(
          side=side,
          type=ORDER_TYPE_MARKET,
          symbol=s,
          quantity=abs(lot))

        if mode == 'LIMIT':
          args['price'] = transactions.price.loc[s]
          args['type'] = ORDER_TYPE_LIMIT
          args['timeInForce'] = 'GTC'

        order_func(**args)
        order_result = 'success'
        print('|', mode, s, side, abs(lot), order_result)
      except Exception as e:
        print('| FAIL', s, s, side, abs(lot), str(e))
        order_result = 'FAIL: ' + str(e)

      trades[s] = {
        **args,
        'result': order_result,
      }

    return pd.DataFrame(trades).transpose()

  def status(self, ohlcvs):
    import ipywidgets as widgets

    ret = pd.DataFrame()
    full_results = []
    for method in self._trading_methods:
      for symbol in method.symbols:
        ohlcv = ohlcvs[(symbol, method.freq)]
        result = method.strategy.backtest(ohlcv,
                method.variables, filters=method.filters, freq=method.freq)
        ret[method.name + '-' + symbol + '-' + method.freq] = result.cumulative_returns
        full_results.append({
          'name': method.name,
          'symbol': symbol,
          'freq': method.freq,
          'weight': method.weight_btc,
          'portfolio': result,
          'trading_method': method,
          'signal': result.cash.iloc[-1] == 0,
        })



    method_dropdown = widgets.Dropdown(options=[m.name + '-' + str(i) for i, m in enumerate(self._trading_methods)])
    symbol_dropdown = widgets.Dropdown(options=[symbol + '-' + freq for symbol, freq in ohlcvs.keys()])
    backtest_btn = widgets.Button(description='status')

    backtest_panel = widgets.Output()
    option_panel = widgets.Output()

    def plotly_df(df):

      # Plot
      fig = px.line()
      for sname, s in df.items():
        fig.add_scatter(x=s.index, y=s.values, name=sname) # Not what is desired - need a line
      # fig.show()

    @backtest_panel.capture(clear_output=True)
    def backtest(_):
      method_id = int(method_dropdown.value.split('-')[-1])
      history_id = tuple(symbol_dropdown.value.split('-'))
      ohlcv = ohlcvs[history_id]
      strategy = self._trading_methods[method_id].strategy
      svars = self._trading_methods[method_id].variables
      filters = self._trading_methods[method_id].filters
      strategy.backtest(ohlcv, variables=svars, filters=filters, freq=history_id[-1], plot=True)

    backtest_btn.on_click(backtest)
    dropdowns = widgets.HBox([method_dropdown, symbol_dropdown, backtest_btn])
    with option_panel:
      plotly_df(ret)
      display(pd.DataFrame(full_results))

    return widgets.VBox([option_panel, dropdowns, backtest_panel])

  def portfolio_backtest(self, ohlcvs, min_freq, quote_assets=['BTC', 'USDT', 'BUSD', 'ETH'], fee=0.002, delay=0):

    # backtest_results
    results = []
    for method in self._trading_methods:
      for symbol in method.symbols:
        ohlcv = ohlcvs[(symbol, method.freq)]
        result = method.strategy.backtest(ohlcv,
                method.variables, filters=method.filters, freq=method.freq)
        results.append({
          'name': method.name,
          'symbol': symbol,
          'freq': method.freq,
          'weight': method.weight_btc,
          'portfolio': result,
          'trading_method': method,
          'signal': result.cash.iloc[-1] == 0,
        })

    results = pd.DataFrame(results)

    import matplotlib.pyplot as plt
    position = {}
    quote_substract = {}
    for index, value in results.transpose().items():
      position[value.loc['name']+'|'+ value.symbol + '|' + value.freq] = (value.portfolio.cash == 0).shift(delay).ffill() * value.weight
    position = pd.DataFrame(position).resample(min_freq).last().ffill()
    position.columns = position.columns.str.split('|').str[1]
    position = position.ffill().fillna(0)
    position = position.groupby(position.columns, axis=1).sum()

    # find quote assets
    quote_asset_col = []
    for symbol in position.columns:
        for q in quote_assets:
            if symbol[-len(q):] == q:
                quote_asset_col.append(q)
                break

    quote_position = position.copy()
    quote_position.columns = quote_asset_col
    quote_position = -quote_position.groupby(quote_position.columns, axis=1).sum()

    # calculate return in usdt
    assets = position.columns.str.split('|').str[0].to_list()

    for i, a in enumerate(assets):
      for q in quote_assets:
        if len(a) > 5 and a[-len(q):] == q:
          assets[i] = a[:-len(q)]

    position.columns = assets
    position = position.groupby(position.columns, axis=1).sum()
    quote_position = quote_position.groupby(quote_position.columns, axis=1).sum()

    all_symbols = list(set(quote_position.columns) | set(position.columns) | set(self._margins.keys()))
    if 'USDT' not in all_symbols:
        all_symbols.append('USDT')

    position = position.reindex(all_symbols, axis=1).fillna(0) + quote_position.reindex(all_symbols, axis=1).fillna(0)

    ohlcv_usdt = {a:get_all_binance(a+'USDT', min_freq) for a in position.columns if a != 'USDT'}

    initial_margin_sum_btc = 0
    for a, w in self._margins.items():
        position[a] += self.ticker_info.get_asset_price_in_btc(a) * w
        initial_margin_sum_btc += self.ticker_info.get_asset_price_in_btc(a) * w

    # remove negative position
    negative_position = ((position < 0) * position).drop('USDT', axis=1, errors='ignore').sum(axis=1)
    pusdt = position['USDT'].copy()
    position = position.clip(0, None)
    position.USDT = pusdt + negative_position

    addition_usdt = -min(position.USDT.min(), 0) / self.ticker_info.get_asset_price_in_btc('USDT')

    if addition_usdt > 0:
        print('WARRN**: additional usdt is required: ', addition_usdt, ' USD')


    p = position.loc[position.index[(position != position.shift()).abs().sum(axis=1) != 0] | position.index[-1:]]
    p.index = p.index.tz_localize(None)

    ohlcv_usdt_close = pd.DataFrame({name:s.close for name, s in ohlcv_usdt.items()})
    ohlcv_usdt_close.index = ohlcv_usdt_close.index.tz_localize(None)

    rebalance_time = (p.index & ohlcv_usdt_close.index)

    ohlcv_usdt_close = ohlcv_usdt_close.loc[rebalance_time]
    p = p.loc[rebalance_time].fillna(0)

    asset_return = ((ohlcv_usdt_close.pct_change().shift(-1).fillna(0)) * p) - fee * (p - p.shift()).abs()
    asset_return.fillna(0, inplace=True)

    (asset_return.cumsum() / self.ticker_info.get_asset_price_in_btc('USDT')).plot()
    plt.show()

    s = (asset_return.sum(axis=1).cumsum() + initial_margin_sum_btc) / self.ticker_info.get_asset_price_in_btc('USDT')
    s.plot()

    plt.show()

    (s / s.cummax()).plot()
    plt.show()

    return results

def render_html(signals_df, rebalance_df, rebalance_df_in_btc, orders, order_results):

  html = """
    <!DOCTYPE html>
    <head>
      <title>Saying Hello</title>
      <link rel="stylesheet" href="https://unpkg.com/purecss@2.0.3/build/pure-min.css" integrity="sha384-cg6SkqEOCV1NbJoCu11+bm0NvBRc8IYLRGXkmNrqUBfTjmMYwNKPWBTIKyw9mHNJ" crossorigin="anonymous">
      <meta name="viewport" content="width=device-width, initial-scale=1">

    </head>
    <body style="padding: 5vw">
    """

  html += '<h1>Crypto Portfolio</h1>'
  html += '<h2>Strategy signals</h2>'
  html += signals_df.to_html(classes="pure-table pure-table-horizontal")
  html += '<h2>Position</h2>'
  html += rebalance_df.to_html(classes="pure-table pure-table-horizontal")
  html += '<h2>Position in BTC</h2>'
  html += rebalance_df_in_btc.to_html(classes="pure-table pure-table-horizontal")
  html += '<h2>Orders</h2>'

  if len(orders) > 0:
      orders['result'] = order_results['result']
      html += orders.to_html(classes="pure-table pure-table-horizontal")
  else:
      html += '<p>None</p>'

  html += '<br>'
  html += '<button onclick="update_position(\'MARKET\')">place market orders</button>'
  html += '<button onclick="update_position(\'LIMIT\')">place limit orders</button>'
  html += '</body>'


  html += """
  <script>
  function update_position(mode) {
    // Redirect to next page
    var next_page = window.location.href.split("?")[0] + "?mode=" + mode
    window.location = next_page;
  }
  </script>
  """
  return html
