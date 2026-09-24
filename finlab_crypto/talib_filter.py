"""Build a filter from one TA-Lib indicator."""
import inspect

from finlab_crypto._talib import TalibIndicator
from finlab_crypto.strategy import Filter


def TalibFilter(talib_function_name, condition, **additional_parameters):
    """Filter whose condition comes from a TA-Lib indicator.

    Args:
      talib_function_name: TA-Lib function name, ex: 'LINEARREG_ANGLE'.
      condition: ``(ohlcv, indicator) -> bool Series``, or
        ``(ohlcv, indicator, params) -> bool Series`` to use
        ``additional_parameters``.
      **additional_parameters: Extra optimisable parameters passed to
        ``condition`` as ``params``, ex: ``threshold=0``.

    Returns:
      Filter: its parameters are the indicator's plus ``additional_parameters``.
    """
    indicator = TalibIndicator(talib_function_name)
    uses_parameters = len(inspect.signature(condition).parameters) > 2

    @Filter(condition=condition, **indicator.parameters, **additional_parameters)
    def talib_filter(ohlcv):
        output = indicator(ohlcv, {name: getattr(talib_filter, name) for name in indicator.parameters})
        if uses_parameters:
            extra = {name: getattr(talib_filter, name) for name in additional_parameters}
            signals = talib_filter.condition(ohlcv, output, extra)
        else:
            signals = talib_filter.condition(ohlcv, output)
        return signals, indicator.figures(output)

    return talib_filter
