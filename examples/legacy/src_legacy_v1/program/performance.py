"""
Performance evaluation and visualization helpers for strategy results.
"""

import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots


# 计算策略评价指标
def strategy_evaluate(equity, select_stock):
    """
    :param equity:  每天的资金曲线
    :param select_stock: 每周期选出的股票
    :return:
    """

    # ===新建一个dataframe保存回测指标
    results = pd.DataFrame()

    # ===计算累积净值
    results.loc[0, '累积净值'] = round(equity['equity_curve'].iloc[-1], 2)

    # ===计算年化收益
    annual_return = (equity['equity_curve'].iloc[-1]) ** (
            '1 days 00:00:00' / (equity['交易日期'].iloc[-1] - equity['交易日期'].iloc[0]) * 365) - 1
    results.loc[0, '年化收益'] = str(round(annual_return * 100, 2)) + '%'

    # ===计算最大回撤，最大回撤的含义：《如何通过3行代码计算最大回撤》https://mp.weixin.qq.com/s/Dwt4lkKR_PEnWRprLlvPVw
    # 计算当日之前的资金曲线的最高点
    equity['max2here'] = equity['equity_curve'].expanding().max()
    # 计算到历史最高值到当日的跌幅，drowdwon
    equity['dd2here'] = equity['equity_curve'] / equity['max2here'] - 1
    # 计算最大回撤，以及最大回撤结束时间
    end_date, max_draw_down = tuple(equity.sort_values(by=['dd2here']).iloc[0][['交易日期', 'dd2here']])
    # 计算最大回撤开始时间
    start_date = equity[equity['交易日期'] <= end_date].sort_values(by='equity_curve', ascending=False).iloc[0][
        '交易日期']
    # 将无关的变量删除
    # equity.drop(['max2here', 'dd2here'], axis=1, inplace=True)
    results.loc[0, '最大回撤'] = format(max_draw_down, '.2%')
    results.loc[0, '最大回撤开始时间'] = str(start_date)
    results.loc[0, '最大回撤结束时间'] = str(end_date)

    # ===年化收益/回撤比：我个人比较关注的一个指标
    results.loc[0, '年化收益/回撤比'] = round(annual_return / abs(max_draw_down), 2)

    # ===统计每个周期
    if not select_stock.empty:
        results.loc[0, '盈利周期数'] = len(select_stock.loc[select_stock['选股下周期涨跌幅'] > 0])  # 盈利笔数
        results.loc[0, '亏损周期数'] = len(select_stock.loc[select_stock['选股下周期涨跌幅'] <= 0])  # 亏损笔数
        results.loc[0, '胜率'] = format(results.loc[0, '盈利周期数'] / len(select_stock), '.2%')  # 胜率
        results.loc[0, '每周期平均收益'] = format(select_stock['选股下周期涨跌幅'].mean(), '.2%')  # 每笔交易平均盈亏
        results.loc[0, '盈亏收益比'] = round(
            select_stock.loc[select_stock['选股下周期涨跌幅'] > 0]['选股下周期涨跌幅'].mean() / \
            select_stock.loc[select_stock['选股下周期涨跌幅'] <= 0]['选股下周期涨跌幅'].mean() * (-1), 2)  # 盈亏比
        results.loc[0, '单周期最大盈利'] = format(select_stock['选股下周期涨跌幅'].max(), '.2%')  # 单笔最大盈利
        results.loc[0, '单周期大亏损'] = format(select_stock['选股下周期涨跌幅'].min(), '.2%')  # 单笔最大亏损

        # ===连续盈利亏损
        results.loc[0, '最大连续盈利周期数'] = max(
            [len(list(v)) for k, v in
             itertools.groupby(np.where(select_stock['选股下周期涨跌幅'] > 0, 1, np.nan))])  # 最大连续盈利次数
        results.loc[0, '最大连续亏损周期数'] = max(
            [len(list(v)) for k, v in
             itertools.groupby(np.where(select_stock['选股下周期涨跌幅'] <= 0, 1, np.nan))])  # 最大连续亏损次数

    # ===每年、每月收益率
    equity.set_index('交易日期', inplace=True)
    year_return = equity[['涨跌幅']].resample(rule='A').apply(lambda x: (1 + x).prod() - 1)
    year_return['指数涨跌幅'] = equity[['指数涨跌幅']].resample(rule='A').apply(lambda x: (1 + x).prod() - 1)
    year_return['超额收益'] = year_return['涨跌幅'] - year_return['指数涨跌幅']

    def num2pct(x):
        return str(round(x * 100, 2)) + '%'

    # 转换成百分比格式
    year_return['涨跌幅'] = year_return['涨跌幅'].apply(num2pct)
    year_return['指数涨跌幅'] = year_return['指数涨跌幅'].apply(num2pct)
    year_return['超额收益'] = year_return['超额收益'].apply(num2pct)
    # 计算月度分析数据
    monthly_return = equity[['涨跌幅']].resample(rule='M').apply(lambda x: (1 + x).prod() - 1)
    monthly_return['指数涨跌幅'] = equity[['指数涨跌幅']].resample(rule='M').apply(lambda x: (1 + x).prod() - 1)
    monthly_return['超额收益'] = monthly_return['涨跌幅'] - monthly_return['指数涨跌幅']

    monthly_return['涨跌幅'] = monthly_return['涨跌幅'].apply(num2pct)
    monthly_return['指数涨跌幅'] = monthly_return['指数涨跌幅'].apply(num2pct)
    monthly_return['超额收益'] = monthly_return['超额收益'].apply(num2pct)

    return results.T, year_return, monthly_return


# 绘制策略曲线
def draw_equity_curve_mat(df, data_dict, date_col=None, right_axis=None, pic_size=[16, 9], dpi=72, font_size=25,
                          log=False, chg=False, title=None, y_label='净值'):
    """
    绘制策略曲线
    :param df: 包含净值数据的df
    :param data_dict: 要展示的数据字典格式：｛图片上显示的名字:df中的列名｝
    :param date_col: 时间列的名字，如果为None将用索引作为时间列
    :param right_axis: 右轴数据 ｛图片上显示的名字:df中的列名｝
    :param pic_size: 图片的尺寸
    :param dpi: 图片的dpi
    :param font_size: 字体大小
    :param chg: datadict中的数据是否为涨跌幅，True表示涨跌幅，False表示净值
    :param log: 是都要算对数收益率
    :param title: 标题
    :param y_label: Y轴的标签
    :return:
    """
    # 复制数据
    draw_df = df.copy()
    # 模块基础设置
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # plt.style.use('dark_background')

    plt.figure(num=1, figsize=(pic_size[0], pic_size[1]), dpi=dpi)
    # 获取时间轴
    if date_col:
        time_data = draw_df[date_col]
    else:
        time_data = draw_df.index
    # 绘制左轴数据
    for key in data_dict:
        if chg:
            draw_df[data_dict[key]] = (draw_df[data_dict[key]] + 1).fillna(1).cumprod()
        if log:
            draw_df[data_dict[key]] = np.log(draw_df[data_dict[key]])
        plt.plot(time_data, draw_df[data_dict[key]], linewidth=2, label=str(key))
    # 设置坐标轴信息等
    plt.ylabel(y_label, fontsize=font_size)
    plt.legend(loc=0, fontsize=font_size)
    plt.tick_params(labelsize=font_size)
    plt.grid()
    if title:
        plt.title(title, fontsize=font_size)

    # 绘制右轴数据
    if right_axis:
        # 生成右轴
        ax_r = plt.twinx()
        # 获取数据
        key = list(right_axis.keys())[0]
        ax_r.plot(time_data, draw_df[right_axis[key]], 'y', linewidth=1, label=str(key))
        # 设置坐标轴信息等
        ax_r.set_ylabel(key, fontsize=font_size)
        ax_r.legend(loc=1, fontsize=font_size)
        ax_r.tick_params(labelsize=font_size)
    plt.show()


def draw_equity_curve_plotly(df, data_dict, date_col=None, right_axis=None, pic_size=[1500, 800], chg=False,
                             title=None, rtn_add=pd.DataFrame()):
    """
    绘制策略曲线
    :param df: 包含净值数据的df
    :param data_dict: 要展示的数据字典格式：｛图片上显示的名字:df中的列名｝
    :param date_col: 时间列的名字，如果为None将用索引作为时间列
    :param right_axis: 右轴数据 ｛图片上显示的名字:df中的列名｝
    :param pic_size: 图片的尺寸
    :param chg: datadict中的数据是否为涨跌幅，True表示涨跌幅，False表示净值
    :param title: 标题
    :param rtn_add: 回测情况
    :return:
    """
    draw_df = df.copy()

    # 设置时间序列
    if date_col:
        time_data = draw_df[date_col]
    else:
        time_data = draw_df.index

    # 绘制左轴数据
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for key in data_dict:
        if chg:
            draw_df[data_dict[key]] = (draw_df[data_dict[key]] + 1).fillna(1).cumprod()
        fig.add_trace(go.Scatter(x=time_data, y=draw_df[data_dict[key]], name=key, ))

    # 绘制右轴数据
    if right_axis:
        key = list(right_axis.keys())[0]
        fig.add_trace(go.Scatter(x=time_data, y=draw_df[right_axis[key]], name=key + '(右轴)',
                                 #  marker=dict(color='rgba(220, 220, 220, 0.8)'),
                                 marker_color='orange', opacity=0.1, line=dict(width=0), fill='tozeroy',
                                 yaxis='y2'))  # 标明设置一个不同于trace1的一个坐标轴
    fig.update_layout(template="none", width=pic_size[0], height=pic_size[1], title_text=title,
                      hovermode="x unified", hoverlabel=dict(bgcolor='rgba(255,255,255,0.5)', ),
                      )
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(label="线性 y轴",
                         method="relayout",
                         args=[{"yaxis.type": "linear"}]),
                    dict(label="Log y轴",
                         method="relayout",
                         args=[{"yaxis.type": "log"}]),
                ])], xaxis=dict(domain=[0.0, 0.73]))

    fig.update_yaxes(
        showspikes=True, spikemode='across', spikesnap='cursor', spikedash='solid', spikethickness=1,  # 峰线
    )
    fig.update_xaxes(
        showspikes=True, spikemode='across+marker', spikesnap='cursor', spikedash='solid', spikethickness=1,  # 峰线
    )
    if not rtn_add.empty:
        # 把rtn放进图里
        rtn_add = rtn_add.T
        rtn_add['最大回撤开始时间'] = rtn_add['最大回撤开始时间'].str.replace('00:00:00', '')
        rtn_add['最大回撤结束时间'] = rtn_add['最大回撤结束时间'].str.replace('00:00:00', '')
        rtn_add = rtn_add.T
        header_list = ['项目', '策略表现'] if rtn_add.shape[1] == 1 else ['项目'] + list(rtn_add.columns)
        rtn_add.reset_index(drop=False, inplace=True)
        table_trace = go.Table(header=dict(values=header_list),
                               cells=dict(values=rtn_add.T.values.tolist()),
                               domain=dict(x=[0.77, 1.0], y=[0.2, 0.82]))
        fig.add_trace(table_trace)
        # 图例调一下位置
        fig.update_layout(
            legend=dict(x=0.8, y=1)
        )
    # 原本有暂存，不用了
    # plot(figure_or_data=fig, filename=path, auto_open=False)
    return_fig = plot(fig, include_plotlyjs=True, output_type='div')
    return return_fig


def draw_bar_plotly(data, x, y, title, text, year_return_add=pd.DataFrame(),
                    pic_size=[1500, 800]):
    fig = make_subplots()

    fig.add_trace(go.Bar(x=data[x], y=data[y], text=data[text]), row=1, col=1)
    fig.update_layout(width=pic_size[0], height=pic_size[1],
                      title_text=title, xaxis=dict(domain=[0.0, 0.73])
                      )
    if not year_return_add.empty:
        year_return_add.reset_index(drop=False, inplace=True)
        year_return_add['交易日期'] = pd.to_datetime(year_return_add['交易日期']).dt.date.astype(str)
        year_return_add['交易日期'] = year_return_add['交易日期'].str.replace('-31', '')
        if year_return_add.shape[1] == 6:  # 说明有择时，表格显示会不够，所以简化一下表头
            year_return_add.rename(columns={'交易日期': '年份', '涨跌幅_(带择时)': '带择时', '指数涨跌幅': '指数',
                                            '超额收益': '超额', '超额收益_(带择时)': '择时超额'}, inplace=True)
        table_trace = go.Table(
            header=dict(values=year_return_add.columns.to_list(), fill=dict(color='white'), line=dict(color='black')),
            cells=dict(values=year_return_add.T.values.tolist(), fill=dict(color='white'), line=dict(color='black')),
            domain=dict(x=[0.74, 1], y=[0.3, 0.9]))
        fig.add_trace(table_trace)
    return_fig = plot(fig, include_plotlyjs=True, output_type='div')

    return return_fig


def robustness_test(data, bins=10, date_col='交易日期', factor_col='复合因子', ret_next='下周期每天涨跌幅',
                    pic_size=[1500, 800], year_return_add=pd.DataFrame()):
    # 分组测试稳定性
    data['复合因子_排名'] = data.groupby(date_col)[factor_col].rank(ascending=True, method='first')
    data['count'] = data.groupby(date_col)[date_col].transform('count')
    df_for_group = data[data['count'] >= bins]
    df_for_group['group'] = df_for_group.groupby(date_col)['复合因子_排名'].transform(
        lambda x: pd.qcut(x, q=bins, labels=range(1, bins + 1), duplicates='drop'))
    df_for_group['下周期收益序列'] = df_for_group[ret_next].apply(lambda x: np.prod(np.array(x) + 1))  # 开盘买入
    group_result = df_for_group.groupby([date_col, 'group'])['下周期收益序列'].mean().to_frame()
    group_result.reset_index('group', inplace=True)
    group_result.group = group_result.group.astype('str')
    # 计算基准资产倍数
    banchmark_result = df_for_group.groupby([date_col])['下周期收益序列'].mean().to_frame()
    banchmark_result['group'] = 'benchmark'
    # 合并结果
    result = pd.concat([group_result, banchmark_result])
    # 计算资产倍数
    result['asset'] = result.groupby('group')['下周期收益序列'].cumprod()
    result['asset'] = result['asset'].apply(lambda x: round(x, 2))

    # 绘制策略分箱柱状图
    fig = draw_bar_plotly(result.loc[result.index == result.index[-2]], x='group', y='asset',
                          title=f'{bins}分箱 资金曲线',
                          text='asset', pic_size=pic_size, year_return_add=year_return_add)
    return fig


def draw_hedge_signal_plotly(df, save_path, title, trade_df, _res_loc, buy_method='开盘', pic_size=[1880, 1000]):
    time_data = df['交易日期']
    # 构建画布左轴
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    # 创建自定义悬停文本
    hover_text = []
    for date, pct_change, open_change in zip(time_data.dt.date.astype(str),
                                             df['涨跌幅'].apply(lambda x: str(round(100 * x, 2)) + '%'),
                                             df[f'{buy_method.replace("价", "")}买入涨跌幅'].apply(
                                                 lambda x: str(round(100 * x, 2)) + '%')):
        hover_text.append(
            f'日期: {date}<br>涨跌幅: {pct_change}<br>{buy_method}买入涨跌幅: {open_change}')

    # 绘制k线图
    fig.add_trace(go.Candlestick(
        x=time_data,
        open=df['开盘价_复权'],  # 字段数据必须是元组、列表、numpy数组、或者pandas的Series数据
        high=df['最高价_复权'],
        low=df['最低价_复权'],
        close=df['收盘价_复权'],
        name='k线',
        increasing_line_color='red',  # 涨的K线颜色
        decreasing_line_color='green',  # 跌的K线颜色
        # text=time_data.dt.date.astype(str)  # 自定义悬停文本把日期加上
        text=hover_text,
    ), row=1, col=1)

    # 更新x轴设置，非交易日在X轴上排除
    date_range = pd.date_range(start=time_data.min(), end=time_data.max(), freq='D')
    miss_dates = date_range[~date_range.isin(time_data)].to_list()
    fig.update_xaxes(rangebreaks=[dict(values=miss_dates)])

    # 标记买卖点的数据，绘制在最后
    mark_point_list = []
    for i in df[(df['买入时间'].notna()) | (df['卖出时间'].notna())].index:
        # 获取买卖点信息
        open_signal = df.loc[i, '买入时间']
        close_signal = df.loc[i, '卖出时间']
        # 只有开仓信号，没有平仓信号
        if pd.notnull(open_signal) and pd.isnull(close_signal):
            signal = open_signal
            # 标记买卖点，在最低价下方标记
            y = df.at[i, '最低价_复权'] * 0.99
        # 没有开仓信号，只有平仓信号
        elif pd.isnull(open_signal) and pd.notnull(close_signal):
            signal = close_signal
            # 标记买卖点，在最高价上方标记
            y = df.at[i, '最高价_复权'] * 1.01
        else:  # 同时有开仓信号和平仓信号
            signal = f'{open_signal}_{close_signal}'
            # 标记买卖点，在最高价上方标记
            y = df.at[i, '最高价_复权'] * 1.01
        mark_point_list.append({
            'x': df.at[i, '交易日期'],
            'y': y,
            'showarrow': True,
            'text': signal,
            'ax': 0,
            'ay': 50 * {'卖出': -1, '买入': 1}[signal],
            'arrowhead': 1 + {'卖出': 0, '买入': 2}[signal],
        })

    # 绘制成交额
    fig.add_trace(go.Bar(x=time_data, y=df['成交额'], name='成交额'), row=2, col=1)

    # 做两个信息表
    res_loc = _res_loc.copy()
    res_loc[['持股累计收益', '次均收益率', '次均收益率(算数平均)', '日均收益率', '日均收益率(算数平均)']] = res_loc[
        ['持股累计收益', '次均收益率', '次均收益率(算数平均)', '日均收益率', '日均收益率(算数平均)']].apply(
        lambda x: str(round(100 * x, 3)) + '%' if isinstance(x, float) else x)
    table_trace = go.Table(header=dict(
        values=[[title.split('_')[1]], [title.split('_')[0]]]),
        cells=dict(
            values=[res_loc.index.to_list()[2:-1], res_loc.to_list()[2:-1]]),
        domain=dict(x=[0.8, 0.95], y=[0.5, 0.9]),
    )
    fig.add_trace(table_trace)

    table_trace = go.Table(header=dict(values=list(['买入日期', '卖出日期', '买入价', '卖出价', '收益率'])),
                           cells=dict(
                               values=[trade_df['买入日期'].dt.date, trade_df['卖出日期'].dt.date, trade_df['买入价'],
                                       trade_df['卖出价'], trade_df['收益率']]),
                           domain=dict(x=[0.75, 1.0], y=[0.1, 0.5]))
    fig.add_trace(table_trace)

    # 更新画布布局，把买卖点标记上
    fig.update_layout(annotations=mark_point_list, template="none", width=pic_size[0], height=pic_size[1],
                      title_text=title, hovermode='x',
                      yaxis=dict(domain=[0.25, 1.0]), xaxis=dict(domain=[0.0, 0.73]),
                      yaxis2=dict(domain=[0.05, 0.25]), xaxis2=dict(domain=[0.0, 0.73]),
                      xaxis_rangeslider_visible=False, )
    fig.update_layout(
        legend=dict(x=0.75, y=1)
    )
    # 保存路径
    save_path = save_path + title + '.html'
    plot(figure_or_data=fig, filename=save_path, auto_open=False)


def merge_html(root_path, fig_list, strategy_file):
    # 创建合并后的网页文件
    os.makedirs(os.path.join(root_path, 'data/pic_file'), exist_ok=True)
    merged_html_file = os.path.join(root_path, f'data/pic_file/{strategy_file}汇总.html')

    # 创建自定义HTML页面，嵌入fig对象的HTML内容
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        .figure-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
    </style>
    </head>
    <body>"""
    for fig in fig_list:
        html_content += f"""
        <div class="figure-container">
            {fig}
        </div>
        """
    html_content += '</body> </html>'

    # 保存自定义HTML页面
    with open(merged_html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    res = os.system('start ' + merged_html_file)
    if res != 0:
        os.system('open ' + merged_html_file)



