import sys
import pandas as pd
import datetime
from binance.enums import *
from finlab_crypto.crawler import get_nbars_binance
from binance.client import Client

class TradingMethod():
  def __init__(self, symbols, freq, lookback, strategy, variables, weight_btc, name=''):
    self.symbols = symbols
    self.freq = freq
    self.lookback = lookback
    self.strategy = strategy
    self.variables = variables
    self.weight_btc = weight_btc
    self.name = name

class TradingPortfolio():
  def __init__(self, binance_key, binance_secret):
    self._client = Client(api_key=binance_key, api_secret=binance_secret)
    self._trading_methods = []

  def register(self, trading_method):
    self._trading_methods.append(trading_method)

  def get_all_symbol_lookback(self):

    symbol_lookbacks = {}
    for method in self._trading_methods:
      for a in method.symbols:
        if (a, method.freq) not in symbol_lookbacks or method.lookback < symbol_lookbacks[(a, method.freq)]:
          symbol_lookbacks[(a, method.freq)] = method.lookback

    return symbol_lookbacks

  def get_ohlcvs(self):

    symbol_lookbacks = self.get_all_symbol_lookback()

    ohlcvs = {}
    for (symbol, freq), lookback in symbol_lookbacks.items():
      ohlcvs[(symbol, freq)] = get_nbars_binance(symbol, freq, lookback, self._client)

    return ohlcvs

  def get_latest_signals(self, ohlcvs):

    ret = []
    for method in self._trading_methods:
      for symbol in method.symbols:
        ohlcv = ohlcvs[(symbol, method.freq)]
        if isinstance(method.strategy, str):
          assert method.strategy == 'buy_and_hold'
          signal = True
        else:
          result = method.strategy.backtest(ohlcv, method.variables, lookback=method.lookback, freq=method.freq)
          signal = result.cash.iloc[-1] == 0
          return_ = 0 if not signal else result.positions.records.iloc[-1]['return']

        if isinstance(method.weight_btc, dict):
          if symbol in method.weight_btc:
            weight = method.weight_btc[symbol]
          else:
            weight = method.weight_btc['default']
        else:
          weight = method.weight_btc

        ret.append({
          'symbol': symbol,
          'freq': method.freq,
          'variables': method.variables,
          'method name': method.name,
          'latest_signal': signal,
          'weight_btc': weight,
          'return': return_,
          'value_in_btc': weight * (1 + return_) * signal,
          'latest_price': ohlcv.close.iloc[-1],

        })

    ret = pd.DataFrame(ret)
    return ret

  def calculate_transaction(self, signals, fixed_position, quote_asset_list, algo_threshold_to_rebalance=0.05, fixed_threshold_to_rebalance=0.01, excluded_assets=None):

    def list_select(list, key, value):
      ret = [l for l in list if l[key] == value]
      if len(ret) == 0:
        return None
      else:
        return ret[0]

    def get_base_asset(symbol):
      sinfo = list_select(exinfo['symbols'], 'symbol', symbol)
      return sinfo['baseAsset']

    def get_quote_asset(symbol):
      sinfo = list_select(exinfo['symbols'], 'symbol', symbol)
      return sinfo['quoteAsset']

    def get_asset_price_in_btc(asset):

      if asset == 'BTC':
        return 1

      ret = list_select(tickers, 'symbol', asset + 'BTC')

      if ret is not None:
        return float(ret['price'])

      ret = list_select(tickers, 'symbol', 'BTC' + asset)
      if ret is not None:
        return 1/float(ret['price'])

      return None

    exinfo = self._client.get_exchange_info()
    info = self._client.get_account()
    tickers = self._client.get_symbol_ticker()

    signals['base_asset'] = signals.symbol.map(get_base_asset)
    signals['quote_asset'] = signals.symbol.map(get_quote_asset)
    signals['base_value_btc'] = signals.latest_signal * signals.value_in_btc
    signals['quote_value_btc'] = (~signals.latest_signal) * signals.weight_btc

    # calculate base and quote assets (in btc term)
    base_asset_value = pd.Series(signals.base_value_btc.values, index=signals.base_asset)
    quote_asset_value = pd.Series(signals.quote_value_btc.values, index=signals.quote_asset)
    base_asset_value = base_asset_value.groupby(level=0).sum()
    quote_asset_value = quote_asset_value.groupby(level=0).sum()

    # get position
    position = pd.Series({i['asset']:i['free'] for i in info['balances'] if float(i['free']) != 0}).astype(float)
    position = position[position.index.str[:2] != 'LD']

    # refine asset index
    all_assets = base_asset_value.index | quote_asset_value.index | position.index

    base_asset_value = base_asset_value.reindex(all_assets).fillna(0)
    quote_asset_value = quote_asset_value.reindex(all_assets).fillna(0)
    position = position.reindex(all_assets).fillna(0)

    # calculate algo value
    algo_value_in_btc = base_asset_value + quote_asset_value
    asset_price_in_btc = position.index.map(get_asset_price_in_btc)
    algo_value = algo_value_in_btc / asset_price_in_btc

    max_algo_value = signals.weight_btc.groupby(signals.base_asset).sum().add(signals.weight_btc.groupby(signals.quote_asset).sum(), fill_value=0).reindex(all_assets).fillna(0)

    # calculate diffierence
    fixed_position = pd.Series(fixed_position).reindex(all_assets).fillna(0)

    diff_value_btc = pd.DataFrame({
      'algo_value': algo_value_in_btc,
      'fixed_position': fixed_position * asset_price_in_btc,
      'present_position': position * asset_price_in_btc,
      'difference': algo_value_in_btc + fixed_position * asset_price_in_btc - position * asset_price_in_btc,
      'max_algo_value': max_algo_value,
      'threshold_to_rebalance': max_algo_value * algo_threshold_to_rebalance + fixed_position * asset_price_in_btc * fixed_threshold_to_rebalance,
    })
    diff_value_btc['rebalance'] = diff_value_btc['difference'].abs() >  diff_value_btc['threshold_to_rebalance']
    diff_value_btc['excluded'] = diff_value_btc.index.isin(excluded_assets)

    diff_value = diff_value_btc.copy()
    diff_value = diff_value.div(asset_price_in_btc, axis=0)
    diff_value.rebalance = diff_value.rebalance != 0
    diff_value.excluded = diff_value.excluded != 0

    # calculate transaction

    rebalance_value_btc = diff_value_btc.rebalance * diff_value_btc.difference * (~diff_value_btc.excluded)
    increase_asset_amount = rebalance_value_btc[rebalance_value_btc > 0]
    decrease_asset_amount = rebalance_value_btc[rebalance_value_btc < 0]

    # assumption: all asset in tickers has liquidity
    print("increase_asset_amount", increase_asset_amount)
    print("decrease_asset_amount", decrease_asset_amount)

    txn_btc = {}

    for nai, ai in increase_asset_amount.items():
      for nad, ad in decrease_asset_amount.items():

        symbol = nad + nai
        amount = min(-ad, ai)

        is_valid = list_select(tickers, 'symbol', symbol) is not None and nai in quote_asset_list
        print(symbol, is_valid)

        if is_valid:
          increase_asset_amount.loc[nai] -= amount
          decrease_asset_amount.loc[nad] += amount
          txn_btc[symbol] = -amount
          break

        symbol = nai + nad
        is_valid = list_select(tickers, 'symbol', symbol) is not None and nad in quote_asset_list
        print(symbol, is_valid, quote_asset_list)

        if is_valid:
          increase_asset_amount.loc[nai] -= amount
          decrease_asset_amount.loc[nad] += amount
          txn_btc[symbol] = amount
          break

    transaction_btc = increase_asset_amount.append(decrease_asset_amount)
    transaction_btc.index = transaction_btc.index + 'USDT'
    if 'USDTUSDT' in transaction_btc.index:
      transaction_btc.pop('USDTUSDT')

    transaction_btc = transaction_btc.append(pd.Series(txn_btc))

    transaction = transaction_btc.to_frame(name='value_in_btc')
    transaction['base_asset'] = transaction.index.map(get_base_asset)
    transaction['quote_asset'] = transaction.index.map(get_quote_asset)
    transaction['value'] = transaction['value_in_btc'] / transaction.base_asset.map(get_asset_price_in_btc)
    transaction['price'] = transaction.index.map(lambda s: list_select(tickers, 'symbol', s)['price'])
    transaction = transaction.groupby(level=0).agg(dict(value_in_btc='sum', value='sum', base_asset='first', quote_asset='first', price='first'))

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
      display(diff_value_btc)
      display(transaction)
      display(verify_assets)
      display(verify)

    # verify diff_value
    def get_filters(exinfo, symbol):
      filters = list_select(exinfo['symbols'], 'symbol', symbol)['filters']
      min_lot_size = list_select(filters, 'filterType', 'LOT_SIZE')['minQty']
      step_size = list_select(filters, 'filterType', 'LOT_SIZE')['stepSize']
      min_notional = list_select(filters, 'filterType', 'MIN_NOTIONAL')['minNotional']
      return {
          'min_lot_size': min_lot_size,
          'step_size': step_size,
          'min_notional': min_notional,
      }

    filters = pd.DataFrame({s: get_filters(exinfo, s) for s in transaction.index}).transpose().astype(float)

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
        transaction['final_value_in_btc'] = diff * transaction.base_asset.map(get_asset_price_in_btc)

    return diff_value, diff_value_btc, transaction


  def execute_trades(self, delta_size, mode='TEST'):
    order_func = self._client.create_order if mode == 'LIVE' else self._client.create_test_order

    print('|---------EXECUTION LOG----------|')
    print('| time: ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    trades = {}
    for s, lot in delta_size.items():

      if lot == 0:
        continue

      side = SIDE_BUY if lot > 0 else SIDE_SELL
      try:
        order = order_func(
            side=side,
            type=ORDER_TYPE_MARKET,
            symbol=s,
            quantity=abs(lot))
        order_result = 'success'
        print('|', mode, s, side, abs(lot), order_result)
      except Exception as e:
        print('| FAIL', s, s, side, abs(lot), str(e))
        order_result = 'FAIL: ' + str(e)

      trades[s] = {
        'mode': mode,
        'side': side,
        'type': 'market',
        'quantity': lot,
        'result': order_result,
      }

    return pd.DataFrame(trades).transpose()


def render_html(signals_df, rebalance_df, rebalance_df_in_btc, order_results):

  rebalance_columns_btc = ['asset', 'signal', 'algo_value_in_btc', 'fixed_value_in_btc', 'target_value_in_btc', 'value_in_btc',  'diff_value_in_btc', 'final_diff_value_in_btc']
  rebalance_columns = ['asset', 'signal', 'algo_value', 'fixed_value', 'target_value', 'value', 'diff_value', 'final_diff_value']
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
  html += '<h2>Rebalance Status in BTC</h2>'
  html += rebalance_df.to_html(classes="pure-table pure-table-horizontal")
  html += '<h2>Rebalance Status</h2>'
  html += rebalance_df_in_btc.to_html(classes="pure-table pure-table-horizontal")
  html += '<h2>Order</h2>'
  html += order_results.to_html(classes="pure-table pure-table-horizontal")
  html += '<br>'
  html += '<button onclick="update_position()">place real orders</button>'
  html += '</body>'


  html += """
  <script>
  function update_position() {
    // Redirect to next page
    var next_page = window.location.href.split("?")[0] + "?mode=LIVE"
    window.location = next_page;
  }
  </script>
  """
  return html
