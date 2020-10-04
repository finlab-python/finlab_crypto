import sys
import pandas as pd
import datetime
from binance.enums import *
from finlab_crypto.crawler import get_nbars_binance
from binance.client import Client

def check_before_rebalance(client, signals, prices, invested_base, fixed_assets, rebalance=False):

  # calculate signal
  signals = pd.Series(signals)

  # calculate position in btc
  position_base_value = (invested_base * signals)

  # calculate position in crypto (algo_size)
  symbols = signals.index.to_list()
  last_price = pd.Series({name: prices[name].close.iloc[-1] for name in symbols}).astype(float)
  algo_size = (position_base_value / last_price)

  # get account balance
  info = client.get_account()
  exinfo = client.get_exchange_info()

  # get account balance
  balance = {i['asset']: (i['free'], i['locked']) for i in info['balances']}


  # calculate account size
  account_size = pd.Series({s:[balance[s[:-3]][0]][0] for s in symbols}).astype(float)

  # calculate default position size
  fixed_size = pd.Series(fixed_assets).reindex(symbols).fillna(0).astype(float)

  # exchange info (for lot filters)
  def list_select(list, key, value):
    return [l for l in list if l[key] == value][0]

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
  filters = pd.DataFrame({s: get_filters(exinfo, s) for s in symbols}).transpose().astype(float)

  min_notional = filters.min_notional
  minimum_lot_size = filters.min_lot_size
  step_size = filters.step_size


  # calculate target size
  target_size = algo_size + fixed_size

  # REBALANCE:
  delta_size = ((target_size - account_size) / step_size).astype(int) * step_size

  # NO REBALANCE:
  same_direction = (
      ((algo_size > 0) & (account_size - fixed_size > minimum_lot_size)) |
      ((algo_size == 0) & (account_size - fixed_size < minimum_lot_size))
  )

  delta_size = delta_size * (1-same_direction)
  delta_size[delta_size.abs() < minimum_lot_size] = 0

  # REBALANCE:
  delta_size = ((target_size - account_size) / step_size).astype(int) * step_size

  # NO REBALANCE:
  if not rebalance:
    same_direction = (
        ((algo_size > 0) & (account_size - fixed_size > minimum_lot_size)) |
        ((algo_size < 0) & (account_size - fixed_size < minimum_lot_size))
    )
    delta_size = delta_size * (1-same_direction)

  # minimim lot size filter
  delta_size[delta_size.abs() < minimum_lot_size] = 0
  delta_size[delta_size.abs() * last_price < min_notional] = 0

  return pd.DataFrame({'algo_size':algo_size, 'algo_size_btc': position_base_value,
  'fixed_size':fixed_assets,
  'account_size':account_size,
  'target_size':target_size,
  'target_size_btc': target_size * last_price,
  'delta_size':delta_size,
  'minimum_lot_size':minimum_lot_size,'last_price': last_price
  })



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

  def calculate_transaction(self, signals, fixed_position, quote_asset_list, algo_threshold_to_rebalance=0.05, fixed_threshold_to_rebalance=0.01):

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

    diff_value = diff_value_btc.copy()
    diff_value = diff_value.div(asset_price_in_btc, axis=0)
    diff_value.rebalance = diff_value.rebalance != 0

    # calculate transaction

    rebalance_value_btc = diff_value_btc.rebalance * diff_value_btc.difference
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

  def calculate_rebalance_position(self, df, fixed_position, quote_assets, rebalance_threshold=0.3):

    exinfo = self._client.get_exchange_info()
    info = self._client.get_account()
    tickers = self._client.get_symbol_ticker()

    def get_price_in_btc(symbol):

      # assume global variables: exinfo, tickers

      sinfo = list_select(exinfo['symbols'], 'symbol', symbol)
      base_asset = sinfo['baseAsset']
      quote_asset = sinfo['quoteAsset']

      if base_asset == 'BTC':
        return 1

      ret = list_select(tickers, 'symbol', base_asset + 'BTC')

      if ret is not None:
        return float(ret['price'])

      ret = list_select(tickers, 'symbol', 'BTC' + base_asset)
      return 1/float(ret['price'])

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


    # format fixed position
    fixed_position_in_symbol = {}
    for asset, value in fixed_position.items():
      symbol = asset + quote_assets[asset] if asset in quote_assets else asset + quote_assets['default']
      fixed_position_in_symbol[symbol] = value

    # get algo value
    algo_value_in_btc = (df['latest_signal'] * df['weight_btc']).groupby(df.symbol).sum()

    # calculate current portfolio
    position = pd.Series({i['asset']:i['free'] for i in info['balances'] if float(i['free']) != 0}).astype(float)
    position = position[position.index.str[:2] != 'LD']
    position = position[position.index != 'USDT']
    position = position.to_frame(name='value')
    position.index.set_names('asset', inplace=True)
    position['symbol'] = position.index.map(lambda a:a+quote_assets[a] if a in quote_assets else a+quote_assets['default'])
    position = position.reset_index().set_index('symbol')


    all_symbols = list(set(position.index) | set(algo_value_in_btc.index))
    position = position.reindex(all_symbols)
    position['asset'] = position.index.map(get_base_asset)
    position['quote_asset'] = position.index.map(get_quote_asset)
    position['value'].fillna(0, inplace=True)



    # add price calculation
    position['asset_price_in_btc'] = position.asset.map(get_asset_price_in_btc)
    position['value_in_btc'] = position.value * position.asset_price_in_btc

    # algo value
    position['algo_value_in_btc'] = algo_value_in_btc.reindex(all_symbols).fillna(0)
    reduce_value = position['algo_value_in_btc'].groupby(position.quote_asset).sum().reindex(position.asset).fillna(0)
    reduce_value.index = position.index
    position.algo_value_in_btc -= reduce_value
    position['algo_value'] = position.algo_value_in_btc / position.asset_price_in_btc

    # algo max value
    position['algo_max_value_in_btc'] = df['weight_btc'].groupby(df.symbol).sum()
    position['algo_max_value'] = position.algo_max_value_in_btc / position.asset_price_in_btc

    # fixed value
    position['fixed_value'] = pd.Series(fixed_position_in_symbol).reindex(all_symbols).fillna(0)
    position['fixed_value_in_btc'] = position.fixed_value * position.asset_price_in_btc

    # target value
    position['target_value'] = position.algo_value + position.fixed_value
    position['target_value_in_btc'] = position.target_value * position.asset_price_in_btc

    # difference
    position['diff_value'] = position.target_value - position.value
    position['diff_value_in_btc'] = position.diff_value * position.asset_price_in_btc

    position['signal'] = df.latest_signal.astype(float).groupby(df.symbol).mean().reindex(all_symbols).fillna(0)
    position['price'] = position.index.map(lambda s: list_select(tickers, 'symbol', s)['price'])


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

    filters = pd.DataFrame({s: get_filters(exinfo, s) for s in position.index}).transpose().astype(float)

    min_notional = filters.min_notional
    minimum_lot_size = filters.min_lot_size
    step_size = filters.step_size

    # rebalance filter:
    diff = position['diff_value']
    diff = diff * (((position.diff_value / position.algo_max_value).abs() > rebalance_threshold) | position.algo_max_value.isnull())

    # step size filter
    diff = round((diff / step_size).astype(int) * step_size, 9)

    # minimum lot filter
    diff[diff.abs() < minimum_lot_size] = 0

    # minimum notional filter
    diff[diff.abs() * position.price.astype(float) < min_notional] = 0

    position['final_diff_value'] = diff
    position['final_diff_value_in_btc'] = diff * position.asset_price_in_btc

    return position


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
