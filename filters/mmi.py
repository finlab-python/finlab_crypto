from finlab_crypto.strategy import Filter

@Filter(side='long', window=400, threshold=0.5)
def mmi_filter(ohlcv):
    
    window = mmi_filter.window
    side = mmi_filter.side
    threshold = mmi_filter.threshold
    
    median = ohlcv.close.rolling(window).median()
    p_gt_m = (ohlcv.close > median)
    yp_gt_m = (ohlcv.close.shift() > median)
    
    mmi = ((p_gt_m & yp_gt_m)).rolling(window).mean() if side == 'long' else \
        ((~p_gt_m & ~yp_gt_m)).rolling(window).mean()
    
    figures = {
        'figures': {
            'mmi': mmi
        }
    }
    
    return mmi > threshold, figures