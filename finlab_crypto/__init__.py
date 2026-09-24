"""Develop, verify and deploy crypto trading strategies."""
import vectorbt as vbt

from finlab_crypto import crawler, onchain
from finlab_crypto.strategy import Filter, Strategy

__version__ = '0.3.0'
__all__ = ['Filter', 'Strategy', 'crawler', 'onchain', 'setup']

DEFAULT_INIT_CASH = 100.0
DEFAULT_FEES = 0.001
DEFAULT_SLIPPAGE = 0.001

vbt.settings.portfolio['init_cash'] = DEFAULT_INIT_CASH
vbt.settings.portfolio['fees'] = DEFAULT_FEES
vbt.settings.portfolio['slippage'] = DEFAULT_SLIPPAGE


def setup():
    """Create ./history/, where downloaded prices are cached."""
    crawler.HISTORY_DIR.mkdir(exist_ok=True)
