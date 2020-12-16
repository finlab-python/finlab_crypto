from pyecharts.globals import CurrentConfig, NotebookType
from pyecharts.charts import Kline, Line, Grid, Bar
CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_LAB

import pyecharts.options as opts
import numpy as np
import pandas as pd
from pyecharts.charts import Candlestick

def chart(dfstock, overlaps=dict(), figures=dict(), markers=dict(), markerlines=[], start_date=None, end_date=None):
    """Backtesting Analysis and optimizer dashboard platform.

    Use pyechart and seaborn module to generate interactive variety charts.

    Args:
      dfstock: A dataframe of trading target data.
      overlaps: A dict of overlaps indicator line setting in figure.
      figures: A dict of information needed for picture drawing.
      markers: A dict of which dfstock index needed to be mark.
      markerlines: A tuple(name, x, y ) in dict of drawing the line connection between entry to exist point.
      start_date: A datetime value of the start of dfstock.
      end_date: A datetime value of the end of dfstock .

    Returns:
      grid_chart: chart display.
      chart_size: A dict of chart's height and width values.

    """
    title = 60
    title_margin_top = 30
    main_chart_height = 300
    margin_left = 50
    vol_chart_height = 50
    sub_figure_height = 60
    width = 800

    dfstock = dfstock.loc[start_date:end_date]

    mark_data = []
    for mark in markers:

        if mark[1] not in dfstock.index:
          continue

        x = np.where(dfstock.index == mark[1])[0][0]
        y = dfstock.high.loc[mark[1]]
        color = '#1d6ff2'
        o = opts.MarkPointItem(coord=[float(x), y], value=mark[0], itemstyle_opts=opts.ItemStyleOpts(color=color))
        mark_data.append(o)

    modified_marklines = []
    for markline in markerlines:
        name, x, y = markline
        if x[0] not in dfstock.index or x[1] not in dfstock.index:
          continue
        xx0 = np.where(dfstock.index == x[0])[0][0]
        xx1 = np.where(dfstock.index == x[1])[0][0]
        x = [float(xx0), float(xx1)]
        modified_marklines.append([
        {
            'name': name,
            'coord': [x[0], y[0]],
            'itemStyle': {'color': '#216dc4'}
        },
        {
            'coord': [x[1], y[1]]
        }
        ])

    #for m in modified_marklines:
    #  print(m.opts)
    #  print('------')

    # mark_data += [
    #     opts.MarkPointItem(type_="max", name="最大值", symbol='rect', symbol_size=[50, 20],
    #                        itemstyle_opts=opts.ItemStyleOpts(color='rgba(0,0,0,0.3)')
    #                       ),
    #     opts.MarkPointItem(type_="min", name="最小值", symbol='rect', symbol_size=[50, 20],
    #                        itemstyle_opts=opts.ItemStyleOpts(color='rgba(0,0,0,0.3)')
    #     )
    # ]

    kline = (
        Kline()
        .add_xaxis(xaxis_data=dfstock.index.astype(str).to_list())
        .add_yaxis(
            series_name="klines",
            y_axis=dfstock[['open', 'close', 'low', 'high']].values.tolist(),
            markpoint_opts=opts.MarkPointOpts(
                data=mark_data
            ),
            markline_opts=opts.MarkLineOpts(
                data=modified_marklines,
                label_opts={'position':'insideMiddleTop', 'show': False}
            ),
            itemstyle_opts=opts.ItemStyleOpts(
                color="#ff6183",
                color0="#58d6ac",
                border_color="#ff6183",
                border_color0="#58d6ac",
            ),
        )
        .set_series_opts()
    )

    #################
    # overlap chart
    #################

    overlap_chart = (
        Line()
        .add_xaxis(xaxis_data=dfstock.index.astype(str).to_list())
    )
    for name, o in overlaps.items():
        overlap_chart.add_yaxis(
            series_name=name,
            y_axis=o.loc[start_date:end_date].to_list(),
            is_smooth=True,
            is_hover_animation=False,
            linestyle_opts=opts.LineStyleOpts(opacity=0.5),
            label_opts=opts.LabelOpts(is_show=False),
        )

    # Bar-1
    bar_1 = (
        Bar()
        .add_xaxis(xaxis_data=dfstock.index.astype(str).to_list())
        .add_yaxis(
            series_name="volume",
            yaxis_data=dfstock.volume.loc[start_date:end_date].to_list(),
            xaxis_index=1,
            yaxis_index=1,
            label_opts=opts.LabelOpts(is_show=False),
            # 改进后在 grid 中 add_js_funcs 后变成如下
            itemstyle_opts=opts.ItemStyleOpts(
                color='rgba(0,0,0,0.2)',
            ),
        )
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(
                type_="category",
                grid_index=1,
                axislabel_opts=opts.LabelOpts(is_show=False),
            ),
            yaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(is_show=False),
            ),
            legend_opts=opts.LegendOpts(is_show=False),
        )
    )


    #################
    # indicators
    #################

    def is_item(item):
        return isinstance(item, pd.Series) or isinstance(item, tuple)

    def item_to_chart(name, item):

        if isinstance(item, pd.Series):
            item_type = 'line'
            series = item.loc[start_date:end_date]
        elif isinstance(item, tuple):
            item_type = item[1]
            series = item[0].loc[start_date:end_date]
        else:
            print('Object type not accept (only pd.Series or tuple)')
            raise

        values = series.to_list()
        index = series.index.astype(str).to_list()

        chart = None
        if item_type == 'line':
            chart = Line()
            chart.add_xaxis(xaxis_data=index)
            chart.add_yaxis(series_name=name,
                y_axis=values,
                is_hover_animation=False,
                #linestyle_opts=opts.LineStyleOpts(width=3, opacity=0.5),
                label_opts=opts.LabelOpts(is_show=False),
            )
        elif item_type == 'bar':
            chart = Bar()
            chart.add_xaxis(xaxis_data=index)
            chart.add_yaxis(
                series_name=name,
                yaxis_data=values,
                #xaxis_index=1,
                #yaxis_index=1,
                label_opts=opts.LabelOpts(is_show=False),
            )

        return chart

    example_charts = []
    for name, graph in figures.items():
        if is_item(graph):
            example_charts.append(item_to_chart(name, graph))
        elif isinstance(graph, dict) or isinstance(graph, pd.DataFrame):
            ys = [item_to_chart(name, subgraph) for name, subgraph in graph.items()]
            for y in ys[1:]:
                ys[0].overlap(y)
            example_charts.append(ys[0])
        else:
            raise Exception('cannot support subfigure type')

    if len(dfstock) <= 500:
        range_start = 0
    else:
        range_start =  95#100 - int(10000/len(dfstock))

    kline.set_global_opts(
            legend_opts=opts.LegendOpts(pos_top='0px', pos_left=str(margin_left)),
            xaxis_opts=opts.AxisOpts(is_scale=True),
            yaxis_opts=opts.AxisOpts(
                is_scale=True,
                splitarea_opts=opts.SplitAreaOpts(
                    is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=0.3)
                ),
                #grid_index=1,
                #split_number=3,
                axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                #axistick_opts=opts.AxisTickOpts(is_show=False),
                splitline_opts=opts.SplitLineOpts(is_show=False),
                axislabel_opts=opts.LabelOpts(is_show=True),
            ),
            datazoom_opts=[
                opts.DataZoomOpts(
                    is_show=False,
                    type_="inside",
                    xaxis_index=list(range(len(example_charts)+2)),
                    range_start=range_start,
                    range_end=100,
                ),
                opts.DataZoomOpts(
                    is_show=True,
                    xaxis_index=list(range(len(example_charts)+2)),
                    type_="slider",
                    pos_top="85%",
                    range_start=range_start,
                    range_end=100,
                ),
            ],
            #title_opts=opts.TitleOpts(title="Kline-DataZoom-inside"),
        )

    # Kline And Line
    overlap_kline_line = kline.overlap(overlap_chart)

    total_height = title + main_chart_height + len(example_charts) * (sub_figure_height + title) + 200

    # Grid Overlap + Bar
    grid_chart = Grid(
        init_opts=opts.InitOpts(
            width=str(width) + 'px',
            height=str(total_height) + 'px',
            animation_opts=opts.AnimationOpts(animation=False),
        )
    )
    grid_chart.add(
        overlap_kline_line,
        grid_opts=opts.GridOpts(pos_top=str(title) + 'px',
                                height=str(main_chart_height) + 'px',
                                pos_left=str(margin_left)+'px', pos_right='0'),
    )

    grid_chart.add(
        bar_1,
        grid_opts=opts.GridOpts(pos_top=str(title+main_chart_height-vol_chart_height) + 'px',
                                height=str(vol_chart_height) + 'px',
                                pos_left=str(margin_left)+'px', pos_right='0'),
    )

    for i, chart in enumerate(example_charts):
        title_pos_top = title + main_chart_height + i * (sub_figure_height + title)
        chart.set_global_opts(
                #title_opts=opts.TitleOpts(name, pos_top=str(title_pos_top+title_margin_top) + 'px'),
                legend_opts=opts.LegendOpts(pos_left=str(margin_left), pos_top=str(title_pos_top+title_margin_top) + 'px'),
            )
        chart_pos_top = title_pos_top + title
        grid_chart.add(
            chart,
            grid_opts=opts.GridOpts(pos_top=str(chart_pos_top) + 'px',
                                    height=str(sub_figure_height) + 'px',
                                    pos_left=str(margin_left)+'px', pos_right='0'
                                   ),
        )
        chart_size = {'height': total_height,  'width': width}
    return grid_chart, chart_size
