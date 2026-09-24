"""Shared plumbing for TalibStrategy and TalibFilter."""
import pandas as pd
from talib import abstract

OVERLAP_GROUP = 'Overlap Studies'


class TalibIndicator:
    """A TA-Lib function evaluated on an OHLCV DataFrame."""

    def __init__(self, name):
        self.function = abstract.Function(name)
        self.name = name

    @property
    def parameters(self):
        return dict(self.function.parameters)

    def __call__(self, ohlcv, parameters):
        return self.function(ohlcv, **parameters)

    def figures(self, output):
        """Overlap studies are drawn on the candles, the rest below them."""
        if self.function.info['group'] != OVERLAP_GROUP:
            return {'figures': {self.name: output}}
        if isinstance(output, pd.DataFrame):
            return {'overlaps': dict(output.items())}
        return {'overlaps': {self.name: output}}
