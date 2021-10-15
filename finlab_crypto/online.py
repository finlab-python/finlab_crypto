import sys
import time
import plotly.express as px
from datetime import timezone
import pandas as pd
import datetime
import warnings
from IPython.display import display
from binance.enums import *
from finlab_crypto.crawler import get_nbars_binance, get_all_binance
from binance.client import Client


class TickerInfo():
    """Ticker basic info.

    Get asset amount and convert price to BTC .

    Attributes:
        client: A Binance client object where api_key, api_secret is required.

    """
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
        """Get base asset data of a given symbol.

        Args:
          symbol: A str of trading target name.

        Returns:
            A str of base asset (ex: 'BTC').
        """
        sinfo = self._list_select(self.exinfo['symbols'], 'symbol', symbol)
        return sinfo['baseAsset']

    def get_quote_asset(self, symbol):
        """Get quote asset data of a given symbol.

        Args:
          symbol: A str of trading target name.

        Returns:
          A float of quote asset.
        """
        sinfo = self._list_select(self.exinfo['symbols'], 'symbol', symbol)
        return sinfo['quoteAsset']

    def get_asset_price_in_btc(self, asset):
        """Convert price to BTC .

        Args:
          asset: A str of asset name (ex: 'ETH').

        Returns:
          A float of price in BTC.
        """
        if asset == 'BTC':
            return 1

        ret = self._list_select(self.tickers, 'symbol', asset + 'BTC')

        if ret is not None:
            return float(ret['price'])

        ret = self._list_select(self.tickers, 'symbol', 'BTC' + asset)
        if ret is not None:
            return 1 / float(ret['price'])

        return None


class TradingMethod():
    """Trading method in online init setting.

    Create trading method object for TradingPortfolio register .

    Attributes:
        symbols: A list of trading pair (ex: ['USDTBTC','ETHBTC']).
        freq: A str of trading time period (ex: '4h').
        lookback: An int of the length of historical data (ex:1000).
        strategy: A function that is your customized strategy (ex:trend_strategy).
        variables: A dict of your customized strategy attributes (ex:dict(name='sma',n1=30,n2=130,),).
        weight_btc: A float of btc for each commodity operation (ex: 0.2).
        filters: A dict that is your customized filter (ex:{}).
        name: A str of your trading method name (ex:'altcoin-trend-hullma').

    """
    def __init__(self, symbols, freq, lookback, strategy, variables, weight_btc=None, weight=None, weight_unit=None, filters=None, name='', execution_price='close'):
        self.symbols = symbols
        self.freq = freq
        self.lookback = lookback
        self.strategy = strategy
        self.variables = variables
        self.weight_btc = weight_btc
        self.weight = weight
        self.weight_unit = weight_unit
        self.filters = filters
        self.name = name
        self.execution_price=execution_price

        if self.weight_btc is None and self.weight is None:
            raise Exception("weight_btc or weight is missing.")

        if self.weight_btc is not None and self.weight is not None:
            raise Exception("weight_btc and weight should not be assigned at the same time")

        if self.weight_btc:
            self.weight = self.weight_btc
            self.weight_unit = 'BTC'

class TradingPortfolio():
    """Connect Binance account.

    The core class to connect Binance  with API, in order to connect account info,
    register strategt.

    Attributes:
        binance_key: A str of is binance authorization key.
        binance_secret: A str of is binance authorization secret.

    """
    def __init__(self, binance_key, binance_secret, execute_before_candle_complete=False):
        self._client = Client(api_key=binance_key, api_secret=binance_secret)
        self._trading_methods = []
        self._margins = {}
        self.ticker_info = TickerInfo(self._client)
        self.default_stable_coin = 'USDT'
        self.execute_before_candle_complete = execute_before_candle_complete

    def set_default_stable_coin(self, token):
        self.default_stable_coin = token

    def register(self, trading_method):
        """Rigister TradingMethod object.
        Args:
          trading_method: A object of TradingMethod().
        """

        if trading_method.execution_price == 'open' and self.execute_before_candle_complete:
            raise Exception("Detect execute_before_candle_complete=True and trading_method.execution_price is open"
                    + "Please set trading_method.execute_before_candle_complete to False"
                    + " and execute live trading right after candles are complete.")

        self._trading_methods.append(trading_method)

    def register_margin(self, asset, weight_btc):
        """Rigister weight_btc as operation amount.
        Args:
          asset: A str of asset name (ex: 'USDT')
          weight_btc: A float of btc for each commodity operation (ex: 0.2)
        """
        self._margins[asset] = weight_btc

    def get_all_symbol_lookback(self):
        """Get all symbol lookback.
        Use in get_ohlcvs(self) function.
        Returns:
            A dict of OHLCV lookback.

        """
        symbol_lookbacks = {}
        addition = {}

        weight_units = set()
        max_lookback = 0

        for method in self._trading_methods:
            weight_units.add((method.weight_unit, method.freq))
            max_lookback = max(max_lookback, method.lookback)
            for a in method.symbols:

                quote_asset = self.ticker_info.get_quote_asset(a)
                base_asset = self.ticker_info.get_base_asset(a)
                if (a, method.freq) not in symbol_lookbacks or method.lookback > symbol_lookbacks[(a, method.freq)]:
                    symbol_lookbacks[(a, method.freq)] = method.lookback

                if base_asset != method.weight_unit:
                    new_symbol = base_asset + method.weight_unit
                    if (new_symbol, method.freq) not in addition or method.lookback > addition[(new_symbol, method.freq)]:
                        addition[(new_symbol, method.freq)] = method.lookback

        for w, f in weight_units:
            if w != 'BTC':
                addition[('BTC' + w, f)] = max_lookback

        # add quote asset historical data
        # for (symbol, freq), lookback in symbol_lookbacks.items():
        #     base_asset = self.ticker_info.get_base_asset(symbol)
        #     if base_asset != self.quote_asset:
        #         new_symbol = base_asset + self.quote_asset
        #         addition[(new_symbol, freq)] = lookback

        return {**symbol_lookbacks, **addition}

    def get_ohlcvs(self):
        """Getting histrical price data through binance api.

        Returns:
            A DataFrame of OHLCV data , the number of data length is lookback.

        """
        symbol_lookbacks = self.get_all_symbol_lookback()

        ohlcvs = {}
        for (symbol, freq), lookback in symbol_lookbacks.items():
            ohlcvs[(symbol, freq)] = get_nbars_binance(symbol, freq, lookback, self._client)

        return ohlcvs

    def get_full_ohlcvs(self):
        """Getting all histrical price data through binance api.

        Returns:
            A DataFrame of OHLCV data for all.

        """
        symbol_lookbacks = self.get_all_symbol_lookback()
        ohlcvs = {}

        for (symbol, freq), lookback in symbol_lookbacks.items():
            ohlcvs[(symbol, freq)] = get_all_binance(symbol, freq)
            time.sleep(3)
        return ohlcvs

    def get_latest_signals(self, ohlcvs, html=False):
        """Get latest signals dataframe.

        Choose which strategy to implement on widgets GUI.

        Args:
          ohlcvs: A dataframe of symbel.
          html: A bool of controlling html generation.

        Returns:
          A dataframe of latest_signals data,
          The last_signals column is bool value of whether to execute the transaction.
          The value_in_btc column is present value of assets.
        """
        ret = []
        for method in self._trading_methods:
            for symbol in method.symbols:
                ohlcv = ohlcvs[(symbol, method.freq)].copy()

                # remove incomplete candle
                if self.execute_before_candle_complete == False and method.execution_price == 'close':
                    t = datetime.datetime.utcnow().replace(tzinfo=timezone.utc)
                    delta_t = ohlcv.index[-1] - ohlcv.index[-2]
                    ohlcv = ohlcv.loc[:t-delta_t]

                htmlname = f'{symbol}-{method.freq}-{method.name}.html' if html else None
                result = method.strategy.backtest(ohlcv,
                                                  method.variables, filters=method.filters, plot=html,
                                                  html=htmlname,
                                                  freq=method.freq, fees=0., slippage=0., execution_price=method.execution_price)

                signal = result.cash().iloc[-1] == 0
                return_ = 0

                # find weight if it is in the nested dictionary
                weight = method.weight
                if isinstance(weight, dict):
                    weight = (weight[symbol]
                        if symbol in weight else weight['default'])

                entry_price = 0
                entry_time = 0
                value_in_btc = 0
                trade_price_type = method.execution_price
                if signal:
                    txn = result.positions().records
                    rds = result.orders().records
                    return_ = ohlcv[trade_price_type].iloc[-1] / rds['price'].iloc[-1] - 1
                    entry_price = rds['price'].iloc[-1]
                    entry_time = ohlcv.index[int(rds.iloc[-1]['idx'])]

                    base_asset = self.ticker_info.get_base_asset(symbol)
                    quote_asset = self.ticker_info.get_quote_asset(symbol)

                    if base_asset != method.weight_unit:
                        quote_symbol = base_asset + method.weight_unit
                        quote_history = ohlcvs[(quote_symbol, method.freq)]
                        quote_asset_price_previous = quote_history[trade_price_type].loc[entry_time]
                        quote_asset_price_now = quote_history[trade_price_type].iloc[-1]
                    else:
                        quote_asset_price_previous = 1
                        quote_asset_price_now = 1

                    if method.weight_unit != 'BTC':
                        btc_quote_price_previous = ohlcvs[('BTC' + method.weight_unit, method.freq)][trade_price_type].loc[entry_time]
                        btc_quote_price_now = ohlcvs[('BTC' + method.weight_unit, method.freq)][trade_price_type].iloc[-1]
                    else:
                        btc_quote_price_previous = 1
                        btc_quote_price_now = 1

                    previous_weight_btc = (weight / btc_quote_price_previous)
                    value_in_btc = previous_weight_btc / quote_asset_price_previous * quote_asset_price_now / (btc_quote_price_now / btc_quote_price_previous)
                    weight_btc = previous_weight_btc
                    previous_price_btc = quote_asset_price_previous  / btc_quote_price_previous
                    amount = previous_weight_btc / previous_price_btc

                else:
                    if method.weight_unit != 'BTC':
                        btc_quote_price_now = ohlcvs[('BTC' + method.weight_unit, method.freq)][trade_price_type].iloc[-1]
                    else:
                        btc_quote_price_now = 1

                    weight_btc = weight / btc_quote_price_now
                    amount = 0

                ret.append({
                    'symbol': symbol,
                    'method name': method.name,
                    'latest_signal': signal,
                    'weight_btc': weight_btc,
                    'freq': method.freq,
                    'return': return_,
                    'amount': amount,
                    'value_in_btc': value_in_btc * signal,
                    'latest_price': ohlcv[trade_price_type].iloc[-1],
                    'entry_price': entry_price,
                    'entry_time': entry_time,
                    'html': htmlname,
                })

        ret = pd.DataFrame(ret)
        return ret

    def calculate_position_size(self, signals, rebalance_threshold=0.03, excluded_assets=list()):
        """Calculate the proportion of asset orders.

        Calculate data is based on latest signals dataframe.

        Args:
          signals: A dataframe of signals.
          rebalance_threshold: A float of rebalance_threshold.
          excluded_assets: A list of asset name which are excluded calculation.

        Returns:
          diff_value: A dataframe of how many assets to deposit for each cryptocurrency.
          diff_value_btc: A dataframe of converting cryptocurrency to BTC.
          transaction: A dataframe of transaction(new order) data.
        """
        if self.default_stable_coin not in excluded_assets:
            excluded_assets.append(self.default_stable_coin)

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
        position = pd.Series({i['asset']: i['free'] for i in self.ticker_info.info['balances']
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
            'difference': (algo_value_in_btc + margin_position * asset_price_in_btc).clip(0,None) - position * asset_price_in_btc,
            'rebalance_threshold': (algo_value_in_btc + margin_position * asset_price_in_btc).abs() * rebalance_threshold,
        })
        diff_value_btc['rebalance'] = diff_value_btc['difference'].abs() > diff_value_btc['rebalance_threshold']
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

        # calculate transactions

        rebalance_value_btc = diff_value_btc.rebalance * diff_value_btc.difference * (~diff_value_btc.excluded)
        increase_asset_amount = rebalance_value_btc[rebalance_value_btc > 0]
        decrease_asset_amount = rebalance_value_btc[rebalance_value_btc < 0]

        diff_value_btc['rebalance'] = diff_value_btc['difference'].abs() > diff_value_btc['rebalance_threshold']
        diff_value['rebalance'] = diff_value_btc.rebalance

        txn_btc = {}

        for nai, ai in increase_asset_amount.items():
            for nad, ad in decrease_asset_amount.items():

                symbol = nad + nai
                amount = min(-ad, ai)

                is_valid = self.ticker_info._list_select(self.ticker_info.tickers, 'symbol',
                                                         symbol) is not None and nai in quote_asset_list

                if is_valid:
                    increase_asset_amount.loc[nai] -= amount
                    decrease_asset_amount.loc[nad] += amount
                    txn_btc[symbol] = -amount
                    continue

                symbol = nai + nad
                is_valid = self.ticker_info._list_select(self.ticker_info.tickers, 'symbol',
                                                         symbol) is not None and nad in quote_asset_list

                if is_valid:
                    increase_asset_amount.loc[nai] -= amount
                    decrease_asset_amount.loc[nad] += amount
                    txn_btc[symbol] = amount
                    continue

        # assumption: self.default_stable_coin can be the quote asset for all alt-coins
        transaction_btc = increase_asset_amount.append(decrease_asset_amount)
        transaction_btc.index = transaction_btc.index + self.default_stable_coin

        if self.default_stable_coin in transaction_btc.index:
            transaction_btc.pop(self.default_stable_coin+self.default_stable_coin)

        transaction_btc = transaction_btc.append(pd.Series(txn_btc))

        transaction = transaction_btc.to_frame(name='value_in_btc')

        transaction['base_asset'] = transaction.index.map(self.ticker_info.get_base_asset)
        transaction['quote_asset'] = transaction.index.map(self.ticker_info.get_quote_asset)
        transaction['value'] = transaction['value_in_btc'] / transaction.base_asset.map(
            self.ticker_info.get_asset_price_in_btc)
        transaction['price'] = transaction.index.map(
            lambda s: self.ticker_info._list_select(self.ticker_info.tickers, 'symbol', s)['price'])
        transaction = transaction.groupby(level=0).agg(
            dict(value_in_btc='sum', value='sum', base_asset='first', quote_asset='first', price='first'))
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
            assert verify[verify.index != self.default_stable_coin].all()
        except:
            print(diff_value_btc)
            print(transaction)
            print(verify_assets)
            print(verify)
            raise Exception("validation fail")

        # filter out orders where base asset is in quote asset list (ex: btcusdt)
        # assumption: base asset should only be paired by one quote asset
        transaction = transaction[~(transaction.base_asset.isin(quote_asset_list) & (
                    transaction.value_in_btc.abs() < diff_value_btc.loc[
                transaction['base_asset']].rebalance_threshold.values))]

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

        filters = pd.DataFrame(
            {s: get_filters(self.ticker_info.exinfo, s) for s in transaction.index}).transpose().astype(float)

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
            transaction['final_value_in_btc'] = diff * transaction.base_asset.map(
                self.ticker_info.get_asset_price_in_btc)
        else:
            transaction = pd.DataFrame(None, columns=['final_value'])

        transaction = transaction[transaction['final_value'] != 0]

        return diff_value, diff_value_btc, transaction

    def execute_orders(self, transactions, mode='TEST'):
        """Execute orders to Binance.

        Execute orders by program order.

        Args:
          transactions: A dataframe which is generated by transaction in calculate_position_size() function result.
          mode: A str of transactions mode, we have 3 method.
              'TEST' is simulation.
              'MARKET' is market order which is transaction at the current latest price.
              'LIMIT' is The transaction is done at the specified price. If the specified price is not touched,
                      the transaction has not been completed.

        Returns:
            A dataframe of trades.
        """
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
        """Strategy list widgets.

        Choose which strategy to implement on widgets GUI.

        Args:
          ohlcvs: A dataframe of symbol.

        Returns:
            widget GUI
        """
        import ipywidgets as widgets

        ret = pd.DataFrame()
        full_results = []
        for method in self._trading_methods:
            for symbol in method.symbols:
                ohlcv = ohlcvs[(symbol, method.freq)]
                result = method.strategy.backtest(ohlcv,
                                                  method.variables, filters=method.filters, freq=method.freq)
                ret[method.name + '-' + symbol + '-' + method.freq] = result.cumulative_returns

                weight_btc = method.weight_btc
                if isinstance(weight_btc, dict):
                    weight_btc = (weight_btc[symbol]
                        if symbol in weight_btc else weight_btc['default'])

                full_results.append({
                    'name': method.name,
                    'symbol': symbol,
                    'freq': method.freq,
                    'weight': weight_btc,
                    'portfolio': result,
                    'trading_method': method,
                    'signal': result.cash().iloc[-1] == 0,
                })

        method_dropdown = widgets.Dropdown(options=[m.name + '-' + str(i) for i, m in enumerate(self._trading_methods)])
        symbol_dropdown = widgets.Dropdown(options=[symbol + '-' + freq for symbol, freq in ohlcvs.keys()])
        backtest_btn = widgets.Button(description='status')

        backtest_panel = widgets.Output()
        option_panel = widgets.Output()

        def plotly_df(df):
            """Display plot.
            """
            # Plot
            fig = px.line()
            for sname, s in df.items():
                fig.add_scatter(x=s.index, y=s.values, name=sname)  # Not what is desired - need a line
            # fig.show()

        @backtest_panel.capture(clear_output=True)
        def backtest(_):
            """Display single strategy backtest result.
            """
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

    def portfolio_backtest(self, ohlcvs, min_freq, quote_assets=['BTC', 'USDT', 'BUSD', 'USDC'], fee=0.002, delay=0):
        """Display portfolio backtest result.

        Calculate overall account asset changes.
        Unit is USD

        Args:
          ohlcvs: A dataframe of symbel.
          min_freq: A str of calculation frequency ex('4h').
          quote_assets: A list of assets name ex(['BTC', 'USDT', 'BUSD', 'ETH']).
          fee: A float of trading fee.
          delay: A int of delayed entry and exit setting.
        Returns:
            widget GUI
        """
        # backtest_results
        results = []
        for method in self._trading_methods:
            for symbol in method.symbols:
                ohlcv = ohlcvs[(symbol, method.freq)]
                result = method.strategy.backtest(ohlcv,
                                                  method.variables, filters=method.filters, freq=method.freq)
                # find weight_btc if it is in the nested dictionary
                weight_btc = method.weight_btc
                if isinstance(weight_btc, dict):
                    weight_btc = (weight_btc[symbol]
                        if symbol in weight_btc else weight_btc['default'])

                results.append({
                    'name': method.name,
                    'symbol': symbol,
                    'freq': method.freq,
                    'weight': weight_btc,
                    'portfolio': result,
                    'trading_method': method,
                    'signal': result.cash().iloc[-1] == 0,
                })

        results = pd.DataFrame(results)

        import matplotlib.pyplot as plt
        position = {}
        quote_substract = {}
        for index, value in results.transpose().items():
            position[value.loc['name'] + '|' + value.symbol + '|' + value.freq] = (value.portfolio.cash() == 0).shift(
                delay).ffill() * value.weight
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

        position = position.reindex(all_symbols, axis=1).fillna(0) + quote_position.reindex(all_symbols, axis=1).fillna(
            0)

        ohlcv_usdt = {a: get_all_binance(a + 'USDT', min_freq) for a in position.columns if a != 'USDT'}

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

        ohlcv_usdt_close = pd.DataFrame({name: s.close for name, s in ohlcv_usdt.items()})
        ohlcv_usdt_close.index = ohlcv_usdt_close.index.tz_localize(None)

        rebalance_time = (p.index & ohlcv_usdt_close.index)

        ohlcv_usdt_close = ohlcv_usdt_close.loc[rebalance_time]
        p = p.loc[rebalance_time].fillna(0)

        asset_return = ((ohlcv_usdt_close.pct_change().shift(-1).fillna(0)) * p) - fee * (p - p.shift()).abs()
        asset_return.fillna(0, inplace=True)

        (asset_return.cumsum() / self.ticker_info.get_asset_price_in_btc('USDT')).plot()
        plt.show()

        s = (asset_return.sum(axis=1).cumsum() + initial_margin_sum_btc) / self.ticker_info.get_asset_price_in_btc(
            'USDT')
        s.plot()

        plt.show()

        (s / s.cummax()).plot()
        plt.show()

        return results


def render_html(signals, position, position_btc, orders, order_results):
    """Render html to google cloud platform.

    Integrate order data into tables that display in html.

    Args:
      signals_df: A dataframe of signals which are generated by TradingPortfolio().get_latest_signals().
      rebalance_df: A dataframe of diff_value which are generated by TradingPortfolio().calculate_position_size().
      rebalance_df_in_btc: A dataframe of diff_value_btc which are generated by TradingPortfolio().calculate_position_size().
      orders: A dataframe of transaction which are generated by TradingPortfolio().calculate_position_size().
      order_results: A dataframe of execute_orders which are generated by TradingPortfolio().execute_orders().

    Returns:
        html
    """
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
    html += signals.to_html(classes="pure-table pure-table-horizontal")
    html += '<h2>Position</h2>'
    html += position.to_html(classes="pure-table pure-table-horizontal")
    html += '<h2>Position in BTC</h2>'
    html += position_btc.to_html(classes="pure-table pure-table-horizontal")
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
