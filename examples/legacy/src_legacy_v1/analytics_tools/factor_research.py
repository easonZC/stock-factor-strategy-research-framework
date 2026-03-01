"""
Entry script for cross-sectional factor research and report generation.
"""

import os
import sys

# Add the root directory of the project to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
import pandas as pd
from joblib import Parallel, delayed
from analytics_tools.helpers import plot_charts as PFun
from program import settings as Cfg
from analytics_tools.helpers import factor_utils as tFun
import warnings

warnings.filterwarnings('ignore')

# =====需要配置的东西=====
factor = 'BB_position' # 你想测试的因子 - 修改为PCA复合因子
factor_from = '布林带位置'  # 想测试的因子来自 \program\因子 中的哪个py文件（涉及因子存储目录结构从哪里载入的问题）
per_oft_list = ['W_0']  # 需要分析的周期
target = '下周期涨跌幅'  # 测试因子与下周期涨跌幅的IC，可以选择其他指标比如夏普率等
need_shift = False  # target这列需不需要shift，如果为True则将target这列向下移动一个周期
multiple_process = True  # True为并行，False为串行；测试：运行内存16G的电脑是能够并行的跑完（刚开始运行读取数据时会很卡），运行内存在16G以下的电脑尽量使用串行


# =====几乎不需要配置的东西=====
bins = 5  # 分箱数 (减少分箱数以避免数据不足的问题)
limit = 20  # 降低限制，某个周期至少有20只票，否则过滤掉这个周期；注意：limit需要大于bins
next_ret = '下周期每天涨跌幅'  # 使用下周期每天涨跌幅画分组持仓走势图
data_folder = Cfg.root_path + f'/data/数据整理/'  # 配置读入数据的文件夹路径
industry_col = '新版申万一级行业名称'  # 配置行业的列名
# 行业名称更改信息，比如：21年之前的一级行业采掘在21年之后更名为煤炭
industry_name_change = {'采掘': '煤炭', '化工': '基础化工', '电气设备': '电力设备', '休闲服务': '社会服务',
                        '纺织服装': '纺织服饰', '商业贸易': '商贸零售'}
market_value = '总市值'  # 配置总市值的列名
b_rate = 1.2 / 10000  # 买入手续费
s_rate = 1.12 / 1000  # 卖出手续费
keep_cols = ['交易日期', '股票代码', factor, target, next_ret, industry_col, market_value]
# 如果target列向下shift1个周期，则更新下target指定的列
if need_shift:
    target = '下周期_' + target
# 创建列表，用来保存各个offset的数据
IC_list = []  # IC数据列表
group_nv_list = []  # 分组净值列表
group_hold_value_list = []  # 分组持仓走势列表
style_corr_list = []  # 风格暴露列表
industry_data_list = []  # 行业分析数据列表
market_value_list = []  # 市值分析数据列表


# =====几乎不需要配置的东西=====


def factor_analysis(per_oft):
    # 记录该offset开始处理的时间
    print(f'offset：{per_oft}，开始处理数据...')
    start_date = datetime.datetime.now()

    # 读入数据
    print(f'Reading factor data from path: {Cfg.factor_path}')
    print(f'Factor: {factor}, Factor from: {factor_from}')
    df = tFun.get_factor_by_period(Cfg.factor_path, per_oft, target, need_shift,
                                   factor_from, keep_cols)
    # 如果返回的数据为空，则跳过该offset继续读取下一个offset的数据
    if df.empty:
        print('Empty dataframe returned')
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 删除必要字段为空的部分
    df = df.dropna(subset=keep_cols, how='any')
    
    # Debug: Print factor information
    print(f"Factor name: {factor}")
    print(f"DataFrame columns: {list(df.columns)}")
    print(f"Factor in columns: {factor in df.columns}")
    
    # 将因子信息转换成float类型
    if factor in df.columns:
        df[factor] = df[factor].astype(float)
    else:
        print(f"Warning: Factor '{factor}' not found in DataFrame columns")
        # Don't return empty, let the offset_grouping function handle it
        # return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # =保留每个周期的股票数量大于limit的日期
    df['当周期股票数'] = df.groupby('交易日期')['交易日期'].transform('count')
    df = df[df['当周期股票数'] > limit].reset_index(drop=True)
    
    # Debug: Print data info before offset_grouping
    print(f"Data shape before grouping: {df.shape}")
    print(f"Sample factor values: {df[factor].head() if factor in df.columns else 'Factor not found'}")

    # 将数据按照交易日期和offset进行分组
    print('Starting offset_grouping...')
    df = tFun.offset_grouping(df, factor, bins)
    print('offset_grouping completed')

    # ===计算这个offset下的IC
    IC = tFun.get_IC(df, factor, target, per_oft)
    # ===计算这个offset下的分组资金曲线、分组持仓走势
    group_nv, group_hold_value = tFun.get_group_nv(df, next_ret, b_rate, s_rate, per_oft)
    # ===计算风格暴露
    style_corr = tFun.get_style_corr(df, factor, per_oft)
    # ===计算行业平均IC以及行业占比
    industry_data = tFun.get_industry_data(df, factor, target, industry_col, industry_name_change)
    # ===计算不同市值分组内的平均IC以及市值占比
    market_value_data = tFun.get_market_value_data(df, factor, target, market_value, bins)

    print(f'per_oft：{per_oft}, 处理数据完成，耗时：{datetime.datetime.now() - start_date}')

    return IC, group_nv, group_hold_value, style_corr, industry_data, market_value_data


print(f'开始进行 {factor} 因子分析...')
s_date = datetime.datetime.now()  # 记录开始时间
if multiple_process:
    result_list = Parallel(Cfg.n_job)(delayed(factor_analysis)(per_oft) for per_oft in per_oft_list)
    # 将返回的数据添加到对应的列表中
    for idx, per_oft in enumerate(per_oft_list):
        if not result_list[idx][0].empty:
            IC_list.append(result_list[idx][0])
            group_nv_list.append(result_list[idx][1])
            group_hold_value_list.append(result_list[idx][2])
            style_corr_list.append(result_list[idx][3])
            industry_data_list.append(result_list[idx][4])
            market_value_list.append(result_list[idx][5])
else:
    for per_oft in per_oft_list:  # 遍历offset进行因子分析
        IC, group_nv, group_hold_value, style_corr, industry_data, market_value_data = factor_analysis(per_oft)
        if not IC.empty:
            # 将返回的数据添加到对应的列表中
            IC_list.append(IC)
            group_nv_list.append(group_nv)
            group_hold_value_list.append(group_hold_value)
            style_corr_list.append(style_corr)
            industry_data_list.append(industry_data)
            market_value_list.append(market_value_data)

# 生成一个包含图的列表，之后的代码每画出一个图都添加到该列表中，最后一起画出图
fig_list = []

print('正在汇总各offset数据并画图...')
start_date = datetime.datetime.now()  # 记录开始时间

# ===计算IC、累计IC以及IC的评价指标
IC, IC_info = tFun.IC_analysis(IC_list)
# =画IC走势图，并将IC图加入到fig_list中，最后一起画图
Rank_fig = PFun.draw_ic_plotly(x=IC['交易日期'], y1=IC['RankIC'], y2=IC['累计RankIC'], title='因子RankIC图',
                               info=IC_info)
fig_list.append(Rank_fig)
# =画IC热力图（年份月份），并将图添加到fig_list中
# 处理IC数据，生成每月的平均IC
IC_month = tFun.get_IC_month(IC)
# 画图并添加
hot_fig = PFun.draw_hot_plotly(x=IC_month.columns, y=IC_month.index, z=IC_month, title='RankIC热力图(行：年份，列：月份)')
fig_list.append(hot_fig)

# ===计算分组资金曲线、分箱图、分组持仓走势
group_curve, group_value, group_hold_value = tFun.group_analysis(group_nv_list, group_hold_value_list)
# =画分组资金曲线...
cols_list = [col for col in group_curve.columns if '第' in col]
group_fig = PFun.draw_line_plotly(x=group_curve['交易日期'], y1=group_curve[cols_list], y2=group_curve['多空净值'],
                                  if_log=True, title='分组资金曲线')
fig_list.append(group_fig)
# =画分箱净值图
group_fig = PFun.draw_bar_plotly(x=group_value['分组'], y=group_value['净值'], title='分组净值')
fig_list.append(group_fig)
# =画分组持仓走势
group_fig = PFun.draw_line_plotly(x=group_hold_value['时间'], y1=group_hold_value[cols_list], update_xticks=True,
                                  if_log=False, title='分组持仓走势')
fig_list.append(group_fig)

# ===计算风格暴露
style_corr = tFun.style_analysis(style_corr_list)
if not style_corr.empty:
    # =画风格暴露图
    style_fig = PFun.draw_bar_plotly(x=style_corr['风格'], y=style_corr['相关系数'], title='因子风格暴露图')
    fig_list.append(style_fig)

# ===行业
# =计算行业平均IC以及行业占比
industry_data = tFun.industry_analysis(industry_data_list, industry_col)
# =画行业分组RankIC
industry_fig1 = PFun.draw_bar_plotly(x=industry_data[industry_col], y=industry_data['RankIC'], title='行业RankIC图')
fig_list.append(industry_fig1)
# =画行业暴露
industry_fig2 = PFun.draw_double_bar_plotly(x=industry_data[industry_col],
                                            y1=industry_data['因子第一组选股在各行业的占比'],
                                            y2=industry_data['因子最后一组选股在各行业的占比'],
                                            title='行业占比（可能会受到行业股票数量的影响）')
fig_list.append(industry_fig2)
# ===市值
# =计算不同市值分组内的平均IC以及市值占比
market_value_data = tFun.market_value_analysis(market_value_list)
# =画市值分组RankIC
market_value_fig1 = PFun.draw_bar_plotly(x=market_value_data['市值分组'], y=market_value_data['RankIC'],
                                         title='市值分组RankIC')
fig_list.append(market_value_fig1)
# =画市值暴露
info = '1-{bins}代表市值从小到大分{bins}组'.format(bins=bins)
market_value_fig2 = PFun.draw_double_bar_plotly(x=market_value_data['市值分组'],
                                                y1=market_value_data['因子第一组选股在各市值分组的占比'],
                                                y2=market_value_data['因子最后一组选股在各市值分组的占比'],
                                                title='市值占比', info=info)
fig_list.append(market_value_fig2)

# ===整合上面所有的图
PFun.merge_html(Cfg.root_path, fig_list=fig_list, strategy_file=f'{factor}因子分析报告')
print(f'汇总数据并画图完成，耗时：{datetime.datetime.now() - start_date}')
print(f'{factor} 因子分析完成，耗时：{datetime.datetime.now() - s_date}')




