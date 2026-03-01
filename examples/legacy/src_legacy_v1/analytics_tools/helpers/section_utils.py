"""
Section-level helper functions used by analysis tool workflows.
"""

from program import settings as Cfg
from joblib import Parallel, delayed
import pandas as pd
import os
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots


def get_file_in_folder(path, file_type, contains=None, filters=[], drop_type=False):
    """
    获取指定文件夹下的文件
    :param path: 文件夹路径
    :param file_type: 文件类型
    :param contains: 需要包含的字符串，默认不含
    :param filters: 字符串中需要过滤掉的内容
    :param drop_type: 是否要保存文件类型
    :return:
    """
    file_list = os.listdir(path)
    file_list = [file for file in file_list if file_type in file]
    if contains:
        file_list = [file for file in file_list if contains in file]
    for con in filters:
        file_list = [file for file in file_list if con not in file]
    if drop_type:
        file_list = [file[:file.rfind('.')] for file in file_list]

    return file_list


def load_back_test_daily_data(cls, path, code):
    base_path = path + f'/基础数据/{code}.pkl'
    df = pd.read_pickle(base_path)
    # if os.path.exists(base_path):
    #     df = pd.read_pickle(base_path)
    # else:
    #     file_list = get_file_in_folder(path + f'/基础数据/', '.pkl', filters=['基础数据'])
    #     df_list = Parallel(Cfg.n_job)(delayed(pd.read_pickle)(path + f'/基础数据/' + code) for code in file_list)
    #     df = pd.concat(df_list, ignore_index=True)
    factors = cls.factors.keys()
    common_cols = ['股票代码', '交易日期', '开盘价', '最高价', '最低价', '收盘价', '成交量', '成交额', '流通市值',
                   '总市值', '沪深300成分股', '上证50成分股',
                   '中证500成分股', '中证1000成分股', '中证2000成分股', '创业板指成分股', '新版申万一级行业名称',
                   '新版申万二级行业名称', '新版申万三级行业名称',
                   '09:35收盘价', '09:45收盘价', '09:55收盘价', '复权因子', '收盘价_复权', '开盘价_复权', '最高价_复权',
                   '最低价_复权']

    for fa in factors:
        factors_path = path + f'/{fa}/{code}.pkl'
        fa_df = pd.read_pickle(factors_path)
        # if os.path.exists(factors_path):
        #     fa_df = pd.read_pickle(factors_path)
        # else:
        #     file_list = get_file_in_folder(path + f'/{fa}/', '.pkl', filters=[fa])
        #     df_list = Parallel(Cfg.n_job)(delayed(pd.read_pickle)(path + f'/{fa}/' + code) for code in file_list)
        #     fa_df = pd.concat(df_list, ignore_index=True)
        # fa_df = pd.read_pickle(factors_path)
        # 如果只筛选指定的列
        if len(cls.factors[fa]) > 0:
            fa_df = fa_df[['交易日期', '股票代码'] + cls.factors[fa]]

        # 对比一下前后列名是否有重复的
        repeat_cols = list(set(df.columns).intersection(set(fa_df.columns)))
        # 要排除掉交易日期和股票代码两列
        repeat_cols = [col for col in repeat_cols if col not in ['股票代码', '交易日期']]
        if len(repeat_cols) > 0:
            for col in repeat_cols:
                if col in common_cols:  # 如果是公共列，则删除
                    fa_df.drop(columns=[col], inplace=True)
                else:
                    print(f'{fa}文件中的{col}列与已经加载的数据重名，程序已经自动退出，请检查因子重名的情况后重新运行')
                    raise Exception(
                        f'{fa}文件中的{col}列与已经加载的数据重名，程序已经自动退出，请检查因子重名的情况后重新运行')
        df = pd.merge(df, fa_df, on=['交易日期', '股票代码'], how='left')
    return df


def read_period_and_offset_file(file_path):
    """
    载入周期offset文件
    """
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, encoding='gbk', parse_dates=['交易日期'], skiprows=1)
        return df
    else:
        print(f'文件{file_path}不存在，请获取period_offset.csv文件后再试')
        raise FileNotFoundError('文件不存在')


# 导入指数
def import_index_data(path, back_trader_start=None, back_trader_end=None):
    """
    从指定位置读入指数数据。指数数据来自于：program_back/构建自己的股票数据库/案例_获取股票最近日K线数据.py
    :param back_trader_end: 回测结束时间
    :param back_trader_start: 回测开始时间
    :param path:
    :return:
    """
    # 导入指数数据
    df_index = pd.read_csv(path, parse_dates=['candle_end_time'], encoding='gbk')
    df_index['指数涨跌幅'] = df_index['close'].pct_change()
    df_index['指数收盘价'] = df_index['close'].copy()
    df_index = df_index[['candle_end_time', '指数涨跌幅', 'open', 'high', 'low', 'close', '指数收盘价']]
    df_index.dropna(subset=['指数涨跌幅'], inplace=True)
    df_index.rename(columns={'candle_end_time': '交易日期'}, inplace=True)
    df_index.sort_values(by=['交易日期'], inplace=True)
    df_index.reset_index(inplace=True, drop=True)

    return df_index


def get_trade_info(_df, open_times, close_times, buy_method):
    """
    获取每一笔的交易信息
    :param _df:算完复权的基础价格数据
    :param open_times:买入日的list
    :param close_times:卖出日的list
    :param buy_method:同config.py中设定买入股票的方法，即在什么时候买入
    :return: df:'买入日期', '卖出日期', '买入价', '卖出价', '收益率',在个股结果中展示
    """

    df = pd.DataFrame(columns=['买入日期', '卖出日期'])
    df['买入日期'] = open_times
    df['卖出日期'] = close_times
    # 买入的价格合并
    df = pd.merge(left=df, right=_df[
        ['交易日期', f'{buy_method.replace("价", "")}价_复权', f'{buy_method.replace("价", "")}价']],
                  left_on='买入日期',
                  right_on='交易日期',
                  how='left')
    # 卖出的价格合并
    df = pd.merge(left=df, right=_df[['交易日期', '收盘价_复权', '收盘价']], left_on='卖出日期', right_on='交易日期',
                  how='left')
    # 展示的买入卖出价为非复权价
    df.rename(columns={f'{buy_method.replace("价", "")}价': '买入价', '收盘价': '卖出价'}, inplace=True)
    # 收益率用复权价算
    df['收益率'] = df['收盘价_复权'] / df[f'{buy_method.replace("价", "")}价_复权'] - 1
    # 将收益率转为为百分比格式
    df['收益率'] = df['收益率'].apply(lambda x: str(round(100 * x, 2)) + '%')
    df = df[['买入日期', '卖出日期', '买入价', '卖出价', '收益率']]
    return df


def check_factor_in_df(df, add_factor_main_list, add_factor_sub_list):
    _add_factor_main_list = []
    _add_factor_sub_list = []
    err_factor = []
    for each_factor in add_factor_main_list:
        if each_factor['因子名称'] == '指数':
            _add_factor_main_list.append(each_factor)
        elif each_factor['因子名称'] in df.columns:
            _add_factor_main_list.append(each_factor)
        else:
            err_factor.append(each_factor['因子名称'])
    for item in add_factor_sub_list:
        factor_names = item['因子名称']
        common_factors = []
        for factor in factor_names:
            if (factor in df.columns) or (factor == '指数'):
                common_factors.append(factor)
            else:
                err_factor.append(factor)
        if common_factors:
            _add_factor_sub_list.append({'因子名称': common_factors, '图形样式': item['图形样式']})
    err_factor = list(set(err_factor))
    if len(err_factor):
        print(f'{"、".join(err_factor)} 因子不存在')
    return _add_factor_main_list, _add_factor_sub_list


def draw_hedge_signal_plotly(df, index_df, save_path, title, trade_df, _res_loc, add_factor_main_list,
                             add_factor_sub_list, color_dict, buy_method='开盘', pic_size=[1880, 1000]):
    # # 主图增加,均为折线图。除指数外，均需要在cal_factors中可被计算出
    # add_factor_main_list = [{'因子名称': '指数', '次坐标轴': True},
    #                         {'因子名称': '5日均线', '次坐标轴': False},
    #                         {'因子名称': '20日均线', '次坐标轴': False}
    #                         ]
    #
    # # 附图增加，一个dict为一个子图
    # # 因子名称的list大于1个值，则会被画在同一个图中，没用次坐标轴概念
    # # 图形样式有且仅有三种选择K线图\柱状图\折线图
    # add_factor_sub_list = [{'因子名称': ['指数'], '图形样式': 'K线图'},
    #                        {'因子名称': ['成交额'], '图形样式': '柱状图'},
    #                        {'因子名称': ['Ret_5'], '图形样式': '折线图'},
    #                        {'因子名称': ['筹码集中度', '价格分位数'], '图形样式': '折线图'},
    #                        ]
    # 随机颜色的列表
    color_list = ['#feb71d', '#dc62af', '#4d50bb', '#f0eb8d', '#018b96', '#e7adea']
    color_i = 0
    for each_factor in add_factor_main_list:
        if each_factor['因子名称'] not in color_dict.keys():
            color_dict[each_factor['因子名称']] = color_list[color_i % len(color_list)]
            color_i += 1
    for each_sub in add_factor_sub_list:
        for each_factor in each_sub['因子名称']:
            if each_factor not in color_dict.keys():
                color_dict[each_factor] = color_list[color_i % len(color_list)]
                color_i += 1

    time_data = df['交易日期']
    # 增加多少个子图
    add_rows = len(add_factor_sub_list)

    # 750是主图，add_rows是子图个数。
    pic_size[1] = max(1000, 750 + add_rows * 250)

    # 主图有没有副轴
    have_secondary_y = any(each_factor.get('次坐标轴', False) for each_factor in add_factor_main_list)

    # 构建画布左轴
    fig = make_subplots(rows=1 + len(add_factor_sub_list), cols=1, shared_xaxes=True,
                        specs=[[{"secondary_y": have_secondary_y}]] + add_rows * [[{"secondary_y": False}]])
    # ===先处理主图主要事项
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
        increasing_line_color='#c13945',  # 涨的K线颜色
        decreasing_line_color='#51b82b',  # 跌的K线颜色
        # text=time_data.dt.date.astype(str)  # 自定义悬停文本把日期加上
        text=hover_text,
    ), row=1, col=1)
    # 绘制主图上其它因子（包括指数或者均线）
    for each_factor in add_factor_main_list:
        if each_factor['因子名称'] == '指数':
            fig.add_trace(
                go.Scatter(
                    x=index_df['交易日期'],
                    y=index_df['指数收盘价'],
                    name='指数',
                    marker_color=color_dict['指数']
                ),
                row=1, col=1, secondary_y=each_factor['次坐标轴']
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=df[each_factor['因子名称']],
                    name=each_factor['因子名称'],
                    marker_color=color_dict[each_factor['因子名称']]
                ),
                row=1, col=1, secondary_y=each_factor['次坐标轴']
            )

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
    # 更新画布布局，把买卖点标记上、把主图的大小调整好
    fig.update_layout(annotations=mark_point_list, template="none", width=pic_size[0], height=pic_size[1],
                      title_text=title, hovermode='x',
                      yaxis=dict(domain=[1 - 750 / pic_size[1], 1.0]), xaxis=dict(domain=[0.0, 0.73]),
                      xaxis_rangeslider_visible=False,
                      )
    # 主图有副轴，就更新
    if have_secondary_y:
        fig.update_layout(yaxis2=dict(domain=[1 - 750 / pic_size[1], 1.0]), xaxis2=dict(domain=[0.0, 0.73]))

    # ==绘制子图
    row = 2  # 1是第一个主图，所以不用管
    # 子图的范围都做算好
    y_domains = [[1 - (1000 + 250 * i) / pic_size[1], 1 - (750 + 250 * i) / pic_size[1]] for i in
                 range(0, add_rows)]
    x_domains = [[0.0, 0.73] for _ in range(0, add_rows)]
    # 做每个子图
    for each_factor in add_factor_sub_list:

        graphicStyle = each_factor['图形样式'].upper()
        for each_sub_factor in each_factor['因子名称']:
            if graphicStyle == '柱状图':
                if each_sub_factor + '_排名' in df.columns:
                    fig.add_trace(go.Bar(x=time_data, y=df[each_sub_factor], name=each_sub_factor,
                                         text=["" if pd.isnull(info) else f"排名 {info}" for info in
                                               df[each_sub_factor + '_排名'].to_list()],
                                         textposition='none',
                                         marker_color=color_dict[each_sub_factor]), row=row, col=1)
                else:
                    fig.add_trace(go.Bar(x=time_data, y=df[each_sub_factor], name=each_sub_factor,
                                         marker_color=color_dict[each_sub_factor]), row=row, col=1)

            elif graphicStyle == '折线图':
                if each_sub_factor + '_排名' in df.columns:
                    fig.add_trace(go.Scatter(x=time_data, y=df[each_sub_factor], name=each_sub_factor,
                                             text=["" if pd.isnull(info) else f"排名 {info}" for info
                                                   in df[each_sub_factor + '_排名'].to_list()],

                                             marker_color=color_dict[each_sub_factor]), row=row, col=1)
                else:
                    fig.add_trace(go.Scatter(x=time_data, y=df[each_sub_factor], name=each_sub_factor,
                                             marker_color=color_dict[each_sub_factor]), row=row, col=1)

            elif graphicStyle == 'K线图':
                fig.add_trace(go.Candlestick(
                    x=index_df['交易日期'],
                    open=index_df['open'],  # 字段数据必须是元组、列表、numpy数组、或者pandas的Series数据
                    high=index_df['high'],
                    low=index_df['low'],
                    close=index_df['close'],
                    name='指数',
                    increasing_line_color='#c13945',  # 涨的K线颜色
                    decreasing_line_color='#51b82b',  # 跌的K线颜色
                ), row=row, col=1)
                fig.update_xaxes(rangeslider_visible=False, row=row, col=1)
        fig.update_yaxes(dict(domain=y_domains[row - 2]), row=row)
        fig.update_xaxes(dict(domain=x_domains[row - 2]), row=row)
        fig.update_yaxes(title_text='、'.join(each_factor['因子名称']) + '因子', row=row, col=1)
        row += 1

    # ==做两个信息表放到旁边
    res_loc = _res_loc.copy()
    res_loc[['累计持股收益', '次均收益率_复利', '次均收益率_单利', '日均收益率_复利', '日均收益率_单利']] = res_loc[
        ['累计持股收益', '次均收益率_复利', '次均收益率_单利', '日均收益率_复利', '日均收益率_单利']].apply(
        lambda x: str(round(100 * x, 3)) + '%' if isinstance(x, float) else x)
    table_trace = go.Table(header=dict(
        values=[[title.split('_')[1]], [title.split('_')[0]]]),
        cells=dict(
            values=[res_loc.index.to_list()[2:-1], res_loc.to_list()[2:-1]]),
        domain=dict(x=[0.85, 1], y=[1 - 500 / pic_size[1], 1 - 100 / pic_size[1]]),
    )
    fig.add_trace(table_trace)
    table_trace = go.Table(header=dict(values=list(['买入日期', '卖出日期', '买入价', '卖出价', '收益率'])),
                           cells=dict(
                               values=[trade_df['买入日期'].dt.date, trade_df['卖出日期'].dt.date, trade_df['买入价'],
                                       trade_df['卖出价'], trade_df['收益率']]),
                           domain=dict(x=[0.75, 1.0], y=[0.1, 1 - 500 / pic_size[1]]))
    fig.add_trace(table_trace)

    # 图例的位置调整
    fig.update_layout(legend=dict(x=0.75, y=1))

    # 保存路径
    save_path = save_path + title + '.html'
    plot(figure_or_data=fig, filename=save_path, auto_open=False)




