"""
Strategy execution/backtest pipeline for selection, evaluation, and reporting.
"""

import gc
import os.path
import sys
import pandas as pd

print("Starting strategy execution pipeline...")

# Add parent directory to sys.path to find the program module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from program import performance as Eva
from program import settings as Cfg
from program import runtime_utils as Fun

print("Core modules imported.")

import warnings

warnings.filterwarnings('ignore')
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_rows', 5000)  # 最多显示数据的行数
# print输出中文表头对齐
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

# =====判断是否是手工运行本程序
manual_mode = True  # 默认手工运行
if len(sys.argv) > 1:
    Cfg.stg_folder = sys.argv[1]  # 外部输入的参数：跑哪个文件夹下的策略
    Cfg.stg_file = sys.argv[2]  # 外出输入的参数：跑哪个策略
    Cfg.run_all_offset = True if sys.argv[3] == 'True' else False  # 外出输入的参数：是否要运行所有的周期
    strategy_params = Fun.str_to_list(sys.argv[4])  # 外出输入的参数：策略参数，字符串格式，需要到策略内部去解析
    manual_mode = False  # 如果有外部输入，即为程序调用，不是手工运行程序
else:
    strategy_params = [[0]]  # 策略参数，字符串格式，需要到策略内部去解析

# 合并不同offset及合并做图用
equity_list = []
fig_list = []

# =====动态导入选股策略
cls = __import__('program.%s.%s' % (Cfg.stg_folder, Cfg.stg_file), fromlist=('',))
print('\n策略文件：', Cfg.stg_file)

# 导入指数数据
index_data = Fun.import_index_data(Cfg.index_path, back_trader_start=Cfg.date_start, back_trader_end=Cfg.date_end)
print(f'从指数获取最新交易日：{index_data["交易日期"].iloc[-1].strftime("%Y-%m-%d")}')

# 加载所有周期所有offset的df
per_oft_df = Fun.read_period_and_offset_file(Cfg.period_offset_file)
print(f'{"回测模式 所有offset都将运行..." if Cfg.run_all_offset else "实盘模式，仅对应offset将被运行"}')

# 入股跑的是实盘，但是选股又有参数，则整个框架会出问题，程序直接退出。
if Cfg.stg_folder == 'live_strategies' and len(strategy_params) > 1:
    print('实盘的选股策略不允许有多个参数，2同一策略如需跑不同的参数，建议拆分多个文件，程序已退出！')
    exit()

# 遍历选股策略的所有offset
for per_oft in cls.period_offset:
    # 判断今日是否需要运行
    run_flag = Fun.judge_current_period(per_oft, index_data, per_oft_df)
    if not Cfg.run_all_offset:  # 判断回测or实盘模式
        # 判断当前offset是否在开仓的状态
        if not run_flag:
            # 不在正确周期
            print(f'实盘模式，周期 & offset：{per_oft}，不予以运行')
            continue

    print(f'\n======== 周期 & offset：{per_oft}========') 

    # =====导入数据
    # 从pickle文件中读取整理好的所有股票数据，包含基础数据和因子数据
    source_df = Fun.load_back_test_data(cls, per_oft, os.path.join(Cfg.factor_path, '周期数据/'))
    # 按照config给出的开始结束时间对数据进行整理
    if Cfg.date_start:
        source_df = source_df[source_df['交易日期'] >= pd.to_datetime(Cfg.date_start)]
    if Cfg.date_end:
        source_df = source_df[source_df['交易日期'] <= pd.to_datetime(Cfg.date_end)]
    # 遍历选股参数，实盘的策略不要有选股参数
    for param in strategy_params:
        df = source_df.copy()
        # =====选股
        # ===过滤股票
        df = cls.filter_stock(df)
        # ===按照策略选股
        df, df_for_group = cls.select_stock(df, cls.select_count, strategy_params)
        # ===记录最近的的选股结果，并在原数据中删除
        # 最新选股
        new_select_stock = df[df['下周期每天涨跌幅'].isna() & (df['交易日期'] == df['交易日期'].max())].copy()
        # 删除数据
        df.dropna(subset=['下周期每天涨跌幅'], inplace=True)
        df_for_group.dropna(subset=['下周期每天涨跌幅'], inplace=True)

        # =====整理选中股票数据
        # ===按照开盘买入的方式，修正选中股票在下周期每天的涨跌幅。
        # 即将下周期每天的涨跌幅中第一天的涨跌幅，改成由开盘买入的涨跌幅
        if Cfg.buy_method != '收盘':
            df[f'下日_{Cfg.buy_method}买入涨跌幅'] = df[f'下日_{Cfg.buy_method}买入涨跌幅'].apply(lambda x: [x])
            df['下周期每天涨跌幅'] = df['下周期每天涨跌幅'].apply(lambda x: x[1:])
            df['下周期每天涨跌幅'] = df[f'下日_{Cfg.buy_method}买入涨跌幅'] + df['下周期每天涨跌幅']
        # 保存文件用于回测分析
        if os.path.exists(os.path.join(Cfg.root_path, r'data/分析目录/待分析/')):
            df.to_csv(os.path.join(Cfg.root_path, f'data/分析目录/待分析/{Cfg.stg_file}_{per_oft}_{cls.select_count}.csv'),
                      encoding='gbk', index=False)

        # 计算手续费
        select_stock = Fun.cal_fee_rate(df, Cfg)

        # 计算下周期整体涨跌幅
        select_stock['选股下周期涨跌幅'] = select_stock['选股下周期每天资金曲线'].apply(lambda x: x[-1] - 1)
        # 计算下周期每天的涨跌幅
        select_stock['选股下周期每天涨跌幅'] = select_stock['选股下周期每天资金曲线'].apply(
            lambda x: list(pd.DataFrame([1] + x).pct_change()[0].iloc[1:]))
        del select_stock['选股下周期每天资金曲线']

        # 为了防止有的周期没有选出股票，创造一个空的df，用于填充不选股的周期
        empty_df = Fun.create_empty_data(index_data, per_oft, per_oft_df)
        # 将选股结果更新到empty_df上
        empty_df.update(select_stock)
        select_stock = empty_df

        # 计算整体资金曲线
        select_stock.reset_index(inplace=True)
        select_stock['资金曲线'] = (select_stock['选股下周期涨跌幅'] + 1).cumprod()

        # =====计算选中股票每天的资金曲线
        #  将选股的资金曲线和大盘指数合并
        equity = pd.merge(left=index_data, right=select_stock[['交易日期', '买入股票代码']], on=['交易日期'], how='left', sort=True)
        # 做些数据整理
        equity['持有股票代码'] = equity['买入股票代码'].shift()
        if not new_select_stock.empty:
            equity = equity[equity['交易日期'] <= new_select_stock['交易日期'].max()]
        equity['持有股票代码'].fillna(method='ffill', inplace=True)
        equity.dropna(subset=['持有股票代码'], inplace=True)
        del equity['买入股票代码']

        # 计算全部资金曲线和基准曲线
        equity['涨跌幅'] = select_stock['选股下周期每天涨跌幅'].sum()
        equity['equity_curve'] = (equity['涨跌幅'] + 1).cumprod()
        equity['benchmark'] = (equity['指数涨跌幅'] + 1).cumprod()

        # =====计算择时
        # 不择时的资金曲线
        equity_not_timing = pd.DataFrame()
        # 不择时的选股结果
        select_stock_not_timing = pd.DataFrame()
        # ===判断策略是否包含择时函数，如果没有则不进行择时
        if hasattr(cls, "timing"):
            # 先备份一下旧的资金曲线 和 选股，后面作图之用
            equity_not_timing = equity.copy()
            select_stock_not_timing = select_stock.copy()

            # 把周期offset的df和equity进行合并，为了timing中可以确定调仓时间
            po_df = per_oft_df[['交易日期', f'{per_oft}']].copy()
            po_df.rename(columns={f'{per_oft}': '周期'}, inplace=True)
            po_df['周期'] = po_df['周期'].abs()
            equity = pd.merge(left=equity, right=po_df, on='交易日期', how='left')
            # 进行资金曲线择时
            signal_df = cls.timing(equity.copy(), strategy_params)  # 调用策略文件的中timing函数
            signal_df['signal'].fillna(method='ffill', inplace=True)  # 产生择时signal
            del equity['周期']
            # 把择时singal并入equity
            equity = pd.merge(left=equity, right=signal_df[['交易日期', 'signal']], on='交易日期', how='left')
            # 今天收盘的信号，明天才可以用
            equity['signal'] = equity['signal'].shift()
            equity['signal'].fillna(method='ffill', inplace=True)
            equity['signal'].fillna(value=1, inplace=True)

            # 根据signal重算资金曲线
            equity.loc[equity['signal'] == 0, '涨跌幅'] = 0
            equity.loc[equity['signal'] == 0, '持有股票代码'] = 'empty'
            equity['equity_curve'] = (equity['涨跌幅'] + 1).cumprod()

            # 根据signal 重算选股
            select_stock = pd.merge(left=select_stock, right=signal_df[['交易日期', 'signal']], on='交易日期', how='left')
            select_stock['signal'].fillna(method='ffill', inplace=True)
            select_stock['signal'].fillna(value=1, inplace=True)
            select_stock = select_stock[select_stock['signal'] == 1]
            select_stock.reset_index(inplace=True)

            signal = signal_df['signal'].iloc[-1]  # 存实盘数据用
        else:
            equity['signal'] = 1
            signal = 1

        # =====将选股策略的历史选股、最新选股，保存到本地文件。供后续使用
        # 保存最新的选股结果
        folder_path = os.path.join(Cfg.root_path, f'data/每日选股/{Cfg.stg_folder}/')
        slt_res_path = os.path.join(folder_path, f'{cls.name}_{per_oft}_{cls.select_count}.csv')
        os.makedirs(folder_path, exist_ok=True)
        Fun.save_select_result(slt_res_path, new_select_stock, per_oft_df, run_flag, Cfg.end_exchange, signal)

        # 保存历史选股结果，供后续轮动策略使用
        folder_path = os.path.join(Cfg.root_path, f'data/回测结果/{Cfg.stg_folder}/')
        os.makedirs(folder_path, exist_ok=True)
        Fun.equity_to_csv(equity, Cfg.stg_file, per_oft, cls.select_count, folder_path)

        # =====计算策略评价指标
        rtn, year_return, month_return = Eva.strategy_evaluate(equity, select_stock)
        # 如果有择时，需要拼一下未择时的东西
        if not equity_not_timing.empty:
            # 计算未择时的绩效
            rtn_not_timing, year_return_not_timing, month_return_not_timing = Eva.strategy_evaluate(equity_not_timing,
                                                                                                    select_stock_not_timing)
            # 合并择时和未择时的数据
            rtn, year_return, month_return = Fun.merge_timing_data(rtn, rtn_not_timing, year_return,
                                                                   year_return_not_timing,
                                                                   month_return, month_return_not_timing)
            equity = pd.merge(left=equity, right=equity_not_timing[['equity_curve']],
                              left_index=True, right_index=True, how='left', suffixes=('', '_not_timing'))
        print(rtn, '\n', year_return)

        # 将计算结果保存一下
        Fun.save_back_test_result(cls, per_oft, param, rtn, year_return, Cfg.root_path)

        # =====画图、分组测试等
        if manual_mode:

            # ===画图
            if not equity_not_timing.empty:
                draw_data_dict = {'策略资金曲线': 'equity_curve_not_timing', '策略资金曲线(带择时)': 'equity_curve',
                                  '基准资金曲线': 'benchmark'}
                right_axis_dict = {'回撤(带择时)': 'dd2here'}
            else:
                draw_data_dict = {'策略资金曲线': 'equity_curve', '基准资金曲线': 'benchmark'}
                right_axis_dict = {'回撤': 'dd2here'}
            # 画图
            fig = Eva.draw_equity_curve_plotly(equity,
                                               title=f'{Cfg.stg_file} 周期 & offset：{per_oft} 持股数量:{cls.select_count} 换仓时间:{Cfg.buy_method}',
                                               data_dict=draw_data_dict,
                                               right_axis=right_axis_dict,
                                               # date_col='交易日期',
                                               rtn_add=rtn)
            fig_list.append(fig)
            # 分组测试稳定性
            fig = Eva.robustness_test(df_for_group, bins=10, year_return_add=year_return)
            fig_list.append(fig)

            # 合并offset的资金曲线用
            equity_list.append(equity[equity.columns[equity.columns.to_series().apply(lambda x: 'equity_curve' in x)]])

    # 回收一下内存
    del source_df
    gc.collect()

if manual_mode and Cfg.run_all_offset:
    # 多offset进行合并
    if len(equity_list) > 1:
        print('\n===================合并所有offset=======================')
        # 多offset合成曲线
        equity, equity_not_timing = Fun.merge_offset(equity_list, index_data)
        rtn, year_return, month_return = Eva.strategy_evaluate(equity, pd.DataFrame())
        # 如果有择时，需要拼一下未择时的东西
        if not equity_not_timing.empty:
            rtn_not_timing, year_return_not_timing, month_return_not_timing = Eva.strategy_evaluate(equity_not_timing,
                                                                                                    pd.DataFrame())
            rtn, year_return, month_return = Fun.merge_timing_data(rtn, rtn_not_timing, year_return,
                                                                   year_return_not_timing, month_return,
                                                                   month_return_not_timing)
            equity = pd.merge(left=equity, right=equity_not_timing[['equity_curve']],
                              left_index=True, right_index=True, how='left', suffixes=('', '_not_timing'))
        print(rtn, '\n', year_return)
        equity = equity.reset_index()

        # ===画图
        if not equity_not_timing.empty:
            draw_data_dict = {'策略资金曲线': 'equity_curve_not_timing', '策略资金曲线(带择时)': 'equity_curve', '基准资金曲线': 'benchmark'}
            right_axis_dict = {'回撤(带择时)': 'dd2here'}
        else:
            draw_data_dict = {'策略资金曲线': 'equity_curve', '基准资金曲线': 'benchmark'}
            right_axis_dict = {'回撤': 'dd2here'}
        fig = Eva.draw_equity_curve_plotly(equity,
                                           title=f'{Cfg.stg_file} 周期 & offset：{cls.period_offset} 持股数量:{cls.select_count} 换仓时间:{Cfg.buy_method}',
                                           data_dict=draw_data_dict, right_axis=right_axis_dict, date_col='交易日期',
                                           rtn_add=rtn)
        fig_list = [fig] + fig_list

    # 储存并打开策略结果html
    Eva.merge_html(Cfg.root_path, fig_list, Cfg.stg_file)





