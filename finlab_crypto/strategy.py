from finlab_crypto.utility import (enumerate_variables, enumerate_signal, 
                                   stop_early, plot_combination, plot_strategy,
                                   variable_visualization
                                  )
import copy
import vectorbt as vbt
import pandas as pd
import matplotlib.pyplot as plt

def Strategy(**default_parameters):

    class Strategy:
        def __init__(self, func):
            self.func = func
            self._variables = None
            self.set_parameters(default_parameters)
    
        def set_parameters(self, variables):
            if variables:
                for key, val in variables.items():
                    setattr(self, key, val)
            self._variables = variables
            
        def show_parameters(self):
            print(self._variables)
    
        def backtest(self, ohlcv, variables, plot=False, lookback=None, **args):
    
            variables_without_stop = copy.copy(variables)
    
            if 'stoploss' in variables_without_stop:
                variables_without_stop.pop('stoploss')
    
            if 'profit_targets' in variables_without_stop:
                variables_without_stop.pop('profit_targets')
            
            ohlcv_lookback = ohlcv.iloc[-lookback:] if lookback is not None else ohlcv
            
            variable_enumerate = enumerate_variables(variables_without_stop)
            entries, exits, fig_data = enumerate_signal(ohlcv_lookback, self, variable_enumerate)
    
            if 'stoploss' in variables:
                entries, exits = stop_early(ohlcv_lookback, 
                                            'stoploss', 
                                            variables['stoploss'], entries, exits)
          
            if 'profit_targets' in variables:
                entries, exits = stop_early(ohlcv_lookback, 
                                            'profit_targets', 
                                            variables['profit_targets'], entries, exits)
    
            entries = entries.squeeze()
            exits = exits.squeeze()
    
            portfolio = vbt.Portfolio.from_signals(
                ohlcv_lookback.close, entries.fillna(False), exits.fillna(False), **args)
            
            if plot and isinstance(entries, pd.Series):
                plot_strategy(ohlcv_lookback, entries, exits, portfolio ,fig_data)
            elif plot:
                plot_combination(portfolio)
                plt.show()
                variable_visualization(portfolio)
    
            return portfolio
    

    def deco(func):
        return Strategy(func)

    return deco