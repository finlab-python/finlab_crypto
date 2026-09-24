"""Build a strategy from one TA-Lib indicator."""
from finlab_crypto._talib import TalibIndicator
from finlab_crypto.strategy import Strategy


def TalibStrategy(talib_function_name, entries, exits):
    """Strategy whose signals come from a TA-Lib indicator.

    Args:
      talib_function_name: TA-Lib function name, ex: 'MACD', 'BBANDS', 'RSI'.
      entries: ``(ohlcv, indicator) -> bool Series``,
        ex: ``lambda ohlcv, macd: macd.macdhist > 0``.
      exits: ``(ohlcv, indicator) -> bool Series``.

    Returns:
      Strategy: its parameters are the indicator's (ex: fastperiod, slowperiod).
    """
    indicator = TalibIndicator(talib_function_name)

    @Strategy(entries=entries, exits=exits, **indicator.parameters)
    def strategy(ohlcv):
        output = indicator(ohlcv, {name: getattr(strategy, name) for name in indicator.parameters})
        return strategy.entries(ohlcv, output), strategy.exits(ohlcv, output), indicator.figures(output)

    return strategy
