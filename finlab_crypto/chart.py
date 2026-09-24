"""Interactive candlestick dashboard built with pyecharts."""
import json

import numpy as np
import pandas as pd
import pyecharts.options as opts
from pyecharts.charts import Bar, Grid, Kline, Line

TITLE_HEIGHT = 60
LEGEND_MARGIN_TOP = 30
MAIN_CHART_HEIGHT = 300
MARGIN_LEFT = 50
VOLUME_CHART_HEIGHT = 50
SUB_FIGURE_HEIGHT = 60
BOTTOM_PADDING = 200
WIDTH = '100%'
DATE_FORMAT = '%Y-%m-%d %H:%M'
ECHARTS_URL = 'https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js'
FULL_RANGE_BARS = 500
ZOOM_START_PERCENT = 95
SLIDER_TOP = '85%'
MARKER_COLOR = '#1d6ff2'
TRADE_LINE_COLOR = '#216dc4'
VOLUME_COLOR = 'rgba(0,0,0,0.2)'
UP_COLOR = '#58d6ac'
DOWN_COLOR = '#ff6183'
K_COLORS = {
    'world': {'increasing_line': UP_COLOR, 'decreasing_line': DOWN_COLOR},
    'taiwan': {'increasing_line': DOWN_COLOR, 'decreasing_line': UP_COLOR},
}


def _labels(index):
    return index.strftime(DATE_FORMAT).to_list()


def _candle_colors(k_colors):
    if isinstance(k_colors, str):
        return K_COLORS[k_colors]
    return {**K_COLORS['world'], **k_colors}


def _position(dfstock, timestamp):
    return float(np.flatnonzero(dfstock.index == timestamp)[0])


def _series_chart(name, item, start_date, end_date):
    series, kind = item if isinstance(item, tuple) else (item, 'line')
    series = series.loc[start_date:end_date]
    index = _labels(series.index)
    values = series.to_list()
    if kind == 'line':
        return (Line().add_xaxis(index)
                .add_yaxis(name, values, is_hover_animation=False, label_opts=opts.LabelOpts(is_show=False)))
    if kind == 'bar':
        return Bar().add_xaxis(index).add_yaxis(name, values, label_opts=opts.LabelOpts(is_show=False))
    raise ValueError(f'figure type should be "line" or "bar", got {kind!r}')


def _figure_chart(name, figure, start_date, end_date):
    if isinstance(figure, (pd.Series, tuple)):
        return _series_chart(name, figure, start_date, end_date)
    if isinstance(figure, (dict, pd.DataFrame)):
        charts = [_series_chart(sub_name, sub, start_date, end_date) for sub_name, sub in figure.items()]
        for other in charts[1:]:
            charts[0].overlap(other)
        return charts[0]
    raise TypeError('a figure should be a pd.Series, a (pd.Series, "line"|"bar") tuple, a dict or a pd.DataFrame')


def chart(dfstock, overlaps=None, figures=None, markers=(), markerlines=(), start_date=None, end_date=None,
          k_colors='world'):
    """Candlesticks with volume, overlay indicators, sub-figures and trades.

    Args:
      dfstock: OHLCV DataFrame.
      overlaps: {name: pd.Series} drawn on top of the candles.
      figures: {name: pd.Series | (pd.Series, 'line'|'bar') | dict | pd.DataFrame} drawn below.
      markers: (label, timestamp) points marked above the candles.
      markerlines: (label, [entry_time, exit_time], [entry_price, exit_price]) trade lines.
      start_date, end_date: Displayed range.
      k_colors: 'world' (green up), 'taiwan' (red up), or
        {'increasing_line': color, 'decreasing_line': color}.

    Returns:
      (pyecharts.charts.Grid, {'height': int, 'width': str})
    """
    overlaps = overlaps or {}
    figures = figures or {}
    dfstock = dfstock.loc[start_date:end_date]
    dates = _labels(dfstock.index)
    colors = _candle_colors(k_colors)

    mark_points = [
        opts.MarkPointItem(coord=[_position(dfstock, time), dfstock.high.loc[time]], value=label,
                           itemstyle_opts=opts.ItemStyleOpts(color=MARKER_COLOR))
        for label, time in markers if time in dfstock.index
    ]
    trade_lines = [
        [{'name': label, 'coord': [_position(dfstock, x[0]), y[0]], 'itemStyle': {'color': TRADE_LINE_COLOR}},
         {'coord': [_position(dfstock, x[1]), y[1]]}]
        for label, x, y in markerlines if x[0] in dfstock.index and x[1] in dfstock.index
    ]

    kline = (
        Kline()
        .add_xaxis(dates)
        .add_yaxis(
            series_name='klines',
            y_axis=dfstock[['open', 'close', 'low', 'high']].values.tolist(),
            markpoint_opts=opts.MarkPointOpts(data=mark_points),
            markline_opts=opts.MarkLineOpts(data=trade_lines,
                                            label_opts={'position': 'insideMiddleTop', 'show': False}),
            itemstyle_opts=opts.ItemStyleOpts(
                color=colors['increasing_line'], color0=colors['decreasing_line'],
                border_color=colors['increasing_line'], border_color0=colors['decreasing_line']),
        )
    )

    overlap_chart = Line().add_xaxis(dates)
    for name, series in overlaps.items():
        overlap_chart.add_yaxis(
            series_name=name, y_axis=series.loc[start_date:end_date].to_list(), is_smooth=True,
            is_hover_animation=False, linestyle_opts=opts.LineStyleOpts(opacity=0.5),
            label_opts=opts.LabelOpts(is_show=False))

    volume = (
        Bar()
        .add_xaxis(dates)
        .add_yaxis(series_name='volume', y_axis=dfstock.volume.to_list(), xaxis_index=1, yaxis_index=1,
                   label_opts=opts.LabelOpts(is_show=False),
                   itemstyle_opts=opts.ItemStyleOpts(color=VOLUME_COLOR))
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(type_='category', grid_index=1, axislabel_opts=opts.LabelOpts(is_show=False)),
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show=False)),
            legend_opts=opts.LegendOpts(is_show=False))
    )

    sub_charts = [_figure_chart(name, figure, start_date, end_date) for name, figure in figures.items()]

    zoom_start = 0 if len(dfstock) <= FULL_RANGE_BARS else ZOOM_START_PERCENT
    zoomed_axes = list(range(len(sub_charts) + 2))
    kline.set_global_opts(
        legend_opts=opts.LegendOpts(pos_top='0px', pos_left=str(MARGIN_LEFT)),
        xaxis_opts=opts.AxisOpts(is_scale=True),
        yaxis_opts=opts.AxisOpts(
            is_scale=True,
            splitarea_opts=opts.SplitAreaOpts(is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=0.3)),
            axisline_opts=opts.AxisLineOpts(is_on_zero=False),
            splitline_opts=opts.SplitLineOpts(is_show=False),
            axislabel_opts=opts.LabelOpts(is_show=True)),
        datazoom_opts=[
            opts.DataZoomOpts(is_show=False, type_='inside', xaxis_index=zoomed_axes,
                              range_start=zoom_start, range_end=100),
            opts.DataZoomOpts(is_show=True, type_='slider', xaxis_index=zoomed_axes, pos_top=SLIDER_TOP,
                              range_start=zoom_start, range_end=100),
        ],
    )

    total_height = (TITLE_HEIGHT + MAIN_CHART_HEIGHT + len(sub_charts) * (SUB_FIGURE_HEIGHT + TITLE_HEIGHT)
                    + BOTTOM_PADDING)
    grid_chart = Grid(init_opts=opts.InitOpts(width=WIDTH, height=f'{total_height}px',
                                              animation_opts=opts.AnimationOpts(animation=False)))

    def grid_opts(top, height):
        return opts.GridOpts(pos_top=f'{top}px', height=f'{height}px', pos_left=f'{MARGIN_LEFT}px', pos_right='0')

    grid_chart.add(kline.overlap(overlap_chart), grid_opts=grid_opts(TITLE_HEIGHT, MAIN_CHART_HEIGHT))
    grid_chart.add(volume, grid_opts=grid_opts(TITLE_HEIGHT + MAIN_CHART_HEIGHT - VOLUME_CHART_HEIGHT,
                                               VOLUME_CHART_HEIGHT))
    for i, sub_chart in enumerate(sub_charts):
        title_top = TITLE_HEIGHT + MAIN_CHART_HEIGHT + i * (SUB_FIGURE_HEIGHT + TITLE_HEIGHT)
        sub_chart.set_global_opts(
            legend_opts=opts.LegendOpts(pos_left=str(MARGIN_LEFT), pos_top=f'{title_top + LEGEND_MARGIN_TOP}px'))
        grid_chart.add(sub_chart, grid_opts=grid_opts(title_top + TITLE_HEIGHT, SUB_FIGURE_HEIGHT))

    return grid_chart, {'height': total_height, 'width': WIDTH}


def to_html(grid_chart, height):
    """Standalone HTML page of a chart, with minified options."""
    options = json.dumps(json.loads(grid_chart.dump_options()), separators=(',', ':'))
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><script src="{ECHARTS_URL}"></script></head>
<body style="margin:0">
<div id="chart" style="width:100%;height:{height}px"></div>
<script>echarts.init(document.getElementById('chart')).setOption({options});</script>
</body>
</html>"""
