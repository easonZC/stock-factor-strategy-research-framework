"""
Core runtime utility functions for data transforms, loading, and orchestration.
"""

import ast
import hashlib
import inspect
import json
import os
import re
import time
import warnings
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN
import random
import numpy as np
import pandas as pd
import requests

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 5000)  # 最多显示数据的行数
from sklearn.linear_model import LinearRegression


def cal_fuquan_price(df, fuquan_type='后复权', method=None):
    """
    用于计算复权价格
    :param df: 必须包含的字段：收盘价，前收盘价，开盘价，最高价，最低价
    :param fuquan_type: ‘前复权’或者‘后复权’
    :return: 最终输出的df中，新增字段：收盘价_复权，开盘价_复权，最高价_复权，最低价_复权
    """

    # 计算复权因子
    df['复权因子'] = (df['收盘价'] / df['前收盘价']).cumprod()

    # 计算前复权、后复权收盘价
    if fuquan_type == '后复权':  # 如果使用后复权方法
        df['收盘价_复权'] = df['复权因子'] * (df.iloc[0]['收盘价'] / df.iloc[0]['复权因子'])
    elif fuquan_type == '前复权':  # 如果使用前复权方法
        df['收盘价_复权'] = df['复权因子'] * (df.iloc[-1]['收盘价'] / df.iloc[-1]['复权因子'])
    else:  # 如果给的复权方法非上述两种标准方法会报错
        raise ValueError('计算复权价时，出现未知的复权类型：%s' % fuquan_type)

    # 计算复权
    df['开盘价_复权'] = df['开盘价'] / df['收盘价'] * df['收盘价_复权']
    df['最高价_复权'] = df['最高价'] / df['收盘价'] * df['收盘价_复权']
    df['最低价_复权'] = df['最低价'] / df['收盘价'] * df['收盘价_复权']
    if method and method != '开盘':
        df[f'{method}_复权'] = df[f'{method}'] / df['收盘价'] * df['收盘价_复权']
    # del df['复权因子']
    return df


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
    df_index['指数涨跌幅'] = df_index['close'].pct_change()  # 计算涨跌幅
    df_index = df_index[['candle_end_time', '指数涨跌幅']]
    df_index.dropna(subset=['指数涨跌幅'], inplace=True)
    df_index.rename(columns={'candle_end_time': '交易日期'}, inplace=True)

    # 保留需要的时间段的数据
    if back_trader_start:
        df_index = df_index[df_index['交易日期'] >= pd.to_datetime(back_trader_start)]
    if back_trader_end:
        df_index = df_index[df_index['交易日期'] <= pd.to_datetime(back_trader_end)]
    # 按时间排序和去除索引
    df_index.sort_values(by=['交易日期'], inplace=True)
    df_index.reset_index(inplace=True, drop=True)

    return df_index


def merge_with_index_data(df, index_data, extra_fill_0_list=[]):
    """
    原始股票数据在不交易的时候没有数据。
    将原始股票数据和指数数据合并，可以补全原始股票数据没有交易的日期。
    :param df: 股票数据
    :param index_data: 指数数据
    :param extra_fill_0_list: 合并时需要填充为0的字段
    :return:
    """
    # ===将股票数据和上证指数合并，结果已经排序
    df = pd.merge(left=df, right=index_data, on='交易日期', how='right', sort=True, indicator=True)

    # ===对开、高、收、低、前收盘价价格进行补全处理
    # 用前一天的收盘价，补全收盘价的空值
    df['收盘价'] = df['收盘价'].ffill()
    # 用收盘价补全开盘价、最高价、最低价的空值
    df['开盘价'] = df['开盘价'].fillna(value=df['收盘价'])
    df['最高价'] = df['最高价'].fillna(value=df['收盘价'])
    df['最低价'] = df['最低价'].fillna(value=df['收盘价'])

    # 如果前面算过复权，复权价也做fillna
    if '收盘价_复权' in df.columns:
        df['收盘价_复权'] = df['收盘价_复权'].ffill()
        for col in ['开盘价_复权', '最高价_复权', '最低价_复权']:
            if col in df.columns:
                df[col] = df[col].fillna(value=df['收盘价_复权'])

    # 补全前收盘价
    df['前收盘价'] = df['前收盘价'].fillna(value=df['收盘价'].shift())

    # ===将停盘时间的某些列，数据填补为0
    fill_0_list = ['成交量', '成交额', '涨跌幅', '开盘买入涨跌幅'] + extra_fill_0_list
    df.loc[:, fill_0_list] = df[fill_0_list].fillna(value=0)

    # ===用前一天的数据，补全其余空值
    df.fillna(method='ffill', inplace=True)

    # ===去除上市之前的数据
    df = df[df['股票代码'].notnull()]

    # ===判断计算当天是否交易
    df['是否交易'] = 1
    df.loc[df['_merge'] == 'right_only', '是否交易'] = 0
    del df['_merge']
    df.reset_index(drop=True, inplace=True)
    return df


def transfer_to_period_data(df, po_df, period_offset, extra_agg_dict={}):
    """
    将日线数据转换为相应的周期数据
    :param df:原始数据
    :param po_df:从period_offset.csv载入的数据
    :param period_offset:转换周期
    :param extra_agg_dict:
    :return:
    """
    # 创造一列用于周期末的时间计算
    df['周期最后交易日'] = df['交易日期']

    # agg_dict是周期内数据整合所必须的字典。数据整合方法包括:first(保留周期内第一条数据)、max(保留周期内最大的数据）、min(保留周期内最小的数据)、sum(周期内所有数据求和)、last(保留最新数据)
    agg_dict = {
        # 必须列
        '周期最后交易日': 'last',
        '股票代码': 'last',
        '股票名称': 'last',
        '是否交易': 'last',

        '开盘价': 'first',
        '最高价': 'max',
        '最低价': 'min',
        '收盘价': 'last',
        '成交额': 'sum',
        '流通市值': 'last',
        '总市值': 'last',
        '新版申万一级行业名称': 'last',
        '上市至今交易天数': 'last',

        '下日_是否交易': 'last',
        '下日_开盘涨停': 'last',
        '下日_是否ST': 'last',
        '下日_是否S': 'last',
        '下日_是否退市': 'last',
        '下日_开盘买入涨跌幅': 'last',
        '复权因子': 'last'
    }
    agg_dict = dict(agg_dict, **extra_agg_dict)
    # ===获取period、offset对应的周期表
    # _group为含负数的原始数据，用于把对应非交易日的涨跌幅设置为0（不太理解的话可以打开period_offset.csv文件看一下）
    po_df['_group'] = po_df[period_offset].copy()
    # group为绝对值后的数据，用于对股票数据做groupby
    po_df['group'] = po_df['_group'].abs().copy()
    df = pd.merge(left=df, right=po_df[['交易日期', 'group', '_group']], on='交易日期', how='left')
    # 为了W53（周五买周三卖）这种有空仓日期的周期，把空仓日的涨跌幅设置为0
    df.loc[df['_group'] < 0, '涨跌幅'] = 0

    # ===对个股数据根据周期offset情况，进行groupby后，得到对应的nD/周线/月线数据
    period_df = df.groupby('group').agg(agg_dict)

    # 计算必须额外数据
    period_df['交易天数'] = df.groupby('group')['是否交易'].sum()
    period_df['市场交易天数'] = df.groupby('group')['是否交易'].count()
    # 计算其他因子
    # 计算周期资金曲线
    period_df['每天涨跌幅'] = df.groupby('group')['涨跌幅'].apply(lambda x: list(x))
    period_df['涨跌幅'] = period_df['复权因子'].pct_change()  # 用复权收盘价计算
    first_ret = (np.array(period_df['每天涨跌幅'].iloc[0]) + 1).prod() - 1  # 第一个持仓周期的复利涨跌幅
    period_df['涨跌幅'] = period_df['涨跌幅'].fillna(value=first_ret)  # pct_change()算的第一天是nan，但是实际是存在涨跌幅的，这里做个修正
    period_df.rename(columns={'周期最后交易日': '交易日期'}, inplace=True)

    # 重置索引
    period_df.reset_index(drop=True, inplace=True)

    # 计算下周期每天涨幅
    period_df['下周期每天涨跌幅'] = period_df['每天涨跌幅'].shift(-1)
    period_df['下周期涨跌幅'] = period_df['涨跌幅'].shift(-1)
    del period_df['每天涨跌幅']
    period_df = period_df[period_df['是否交易'] == 1]

    return period_df


# 计算涨跌停
def cal_zdt_price(df):
    """
    计算股票当天的涨跌停价格。在计算涨跌停价格的时候，按照严格的四舍五入。
    包含st股，但是不包含新股
    涨跌停制度规则:
        ---2020年8月23日
        非ST股票 10%
        ST股票 5%

        ---2020年8月24日至今
        普通非ST股票 10%
        普通ST股票 5%

        科创板（sh68） 20%（一直是20%，不受时间限制）
        创业板（sz30） 20%
        科创板和创业板即使ST，涨跌幅限制也是20%

        北交所（bj） 30%

    :param df: 必须得是日线数据。必须包含的字段：前收盘价，开盘价，最高价，最低价
    :return:
    """
    # 计算涨停价格
    # 普通股票
    cond = df['股票名称'].str.contains('ST')
    df['涨停价'] = df['前收盘价'] * 1.1
    df['跌停价'] = df['前收盘价'] * 0.9
    df.loc[cond, '涨停价'] = df['前收盘价'] * 1.05
    df.loc[cond, '跌停价'] = df['前收盘价'] * 0.95

    # 科创板 20%
    rule_kcb = df['股票代码'].str.contains('sh68')
    # 2020年8月23日之后涨跌停规则有所改变
    # 新规的创业板
    new_rule_cyb = (df['交易日期'] > pd.to_datetime('2020-08-23')) & df['股票代码'].str.contains('sz30')
    # 北交所条件
    cond_bj = df['股票代码'].str.contains('bj')

    # 科创板 & 创业板
    df.loc[rule_kcb | new_rule_cyb, '涨停价'] = df['前收盘价'] * 1.2
    df.loc[rule_kcb | new_rule_cyb, '跌停价'] = df['前收盘价'] * 0.8

    # 北交所
    df.loc[cond_bj, '涨停价'] = df['前收盘价'] * 1.3
    df.loc[cond_bj, '跌停价'] = df['前收盘价'] * 0.7

    # 四舍五入
    price_round = lambda x: float(Decimal(x + 1e-7).quantize(Decimal('1.00'), ROUND_HALF_UP))
    df.loc[~cond_bj, '涨停价'] = df['涨停价'].apply(price_round)
    df.loc[~cond_bj, '跌停价'] = df['跌停价'].apply(price_round)

    # 北交所特俗处理：北交所的规则是涨跌停价格小于等于30%，不做四舍五入,所以超过30%的部分需要减去1分钱
    price_round_bj = lambda x: float(Decimal(x).quantize(Decimal('0.00'), rounding=ROUND_DOWN))
    df.loc[cond_bj, '涨停价'] = df['涨停价'].apply(price_round_bj)
    df.loc[cond_bj, '跌停价'] = df['跌停价'].apply(price_round_bj)

    # 判断是否一字涨停
    df['一字涨停'] = False
    df.loc[df['最低价'] >= df['涨停价'], '一字涨停'] = True
    # 判断是否一字跌停
    df['一字跌停'] = False
    df.loc[df['最高价'] <= df['跌停价'], '一字跌停'] = True
    # 判断是否开盘涨停
    df['开盘涨停'] = False
    df.loc[df['开盘价'] >= df['涨停价'], '开盘涨停'] = True
    # 判断是否开盘跌停
    df['开盘跌停'] = False
    df.loc[df['开盘价'] <= df['跌停价'], '开盘跌停'] = True

    return df


def create_empty_data(index_data, period_offset, po_df):
    """
    按照格式和需要的字段创建一个空的dataframe，用于填充不选股的周期

    :param index_data: 指数数据
    :param period_offset: 给定offset周期
    :param po_df: 包含全部offset的dataframe
    """
    empty_df = index_data[['交易日期']].copy()
    empty_df['涨跌幅'] = 0.0
    empty_df['周期最后交易日'] = empty_df['交易日期']
    agg_dict = {'周期最后交易日': 'last'}
    po_df['group'] = po_df[f'{period_offset}'].abs().copy()
    group = po_df[['交易日期', 'group']].copy()
    empty_df = pd.merge(left=empty_df, right=group, on='交易日期', how='left')
    empty_period_df = empty_df.groupby('group').agg(agg_dict)
    empty_period_df['每天涨跌幅'] = empty_df.groupby('group')['涨跌幅'].apply(lambda x: list(x))
    # 删除没交易的日期
    empty_period_df.dropna(subset=['周期最后交易日'], inplace=True)

    empty_period_df['选股下周期每天涨跌幅'] = empty_period_df['每天涨跌幅'].shift(-1)
    empty_period_df.dropna(subset=['选股下周期每天涨跌幅'], inplace=True)

    # 填仓其他列
    empty_period_df['股票数量'] = 0
    empty_period_df['买入股票代码'] = 'empty'
    empty_period_df['买入股票名称'] = 'empty'
    empty_period_df['选股下周期涨跌幅'] = 0.0
    empty_period_df.rename(columns={'周期最后交易日': '交易日期'}, inplace=True)

    empty_period_df.set_index('交易日期', inplace=True)

    empty_period_df = empty_period_df[
        ['股票数量', '买入股票代码', '买入股票名称', '选股下周期涨跌幅', '选股下周期每天涨跌幅']]
    return empty_period_df


def equity_to_csv(equity, strategy_name, period_offset, select_stock_num, folder_path):
    """
    输出策略轮动对应的文件
    :param equity: 策略资金曲线
    :param strategy_name: 策略名称
    :param period_offset: 周期 以及 offset
    :param select_stock_num: 选股数
    :param folder_path: 输出路径
    :return:
    """
    name_str = f'{strategy_name}_{period_offset}_{select_stock_num}'
    to_csv_path = os.path.join(folder_path, f'{name_str}.csv')
    equity['策略名称'] = name_str
    pd.DataFrame(columns=['对数据字段有疑问的，可以直接私信']).to_csv(
        to_csv_path,
        encoding='gbk',
        index=False)
    equity = equity[['交易日期', '策略名称', '持有股票代码', '涨跌幅', 'equity_curve', '指数涨跌幅', 'benchmark']]
    equity.to_csv(to_csv_path, encoding='gbk', index=False, mode='a')


def save_select_result(save_path, new_res, po_df, is_open, end_exchange, signal=1):
    """
    保存最新的选股结果
    :param save_path: 保存路径
    :param new_res: 最新的选股结果
    :param po_df: 周期表
    :param is_open: 今日是否开仓
    :param end_exchange: 是否尾盘换仓
    :param signal: 择时信号
    :return:
    """
    if is_open:  # 仅在正确的周期offset情况下保存实盘文件
        # 获取一下选股日期和交易日期
        select_date = new_res['交易日期'].max()
        # 如果选股结果为空，提示
        if new_res.empty:
            print('当前选股结果为空，可能是策略空仓，或者指数截止日期比股票截止日期多一天导致的，请检查数据。')
            return
        if end_exchange:  # 尾盘换仓的交易日期等于选股日期
            trade_date = select_date
        else:  # 正常情况的交易日期等于选股日期的下一个交易日
            select_inx = po_df[po_df['交易日期'] == select_date].index.min()
            trade_date = po_df['交易日期'].iloc[select_inx + 1]

        # 保存最新的选股结果
        new_res['选股日期'] = select_date
        new_res['交易日期'] = trade_date
        new_res = new_res[['选股日期', '交易日期', '股票代码', '股票名称', '选股排名']]
        if signal != 1:
            new_res = pd.DataFrame(columns=['选股日期', '交易日期', '股票代码', '股票名称', '选股排名'])

        # 申明历史选股结果的变量
        res_df = pd.DataFrame()
        # 如果有历史结果，则读取历史结果
        if os.path.exists(save_path):
            res_df = pd.read_csv(save_path, encoding='gbk', parse_dates=['选股日期', '交易日期'])
            # 有新产生持仓，就把历史结果里的相同日期去掉
            if not new_res.empty:
                res_df = res_df[res_df['选股日期'] < new_res['选股日期'].min()]
                res_df = res_df[res_df['交易日期'] < new_res['交易日期'].min()]

        # 将历史选股结果与最新选股结果合并
        res_df = pd.concat([res_df, new_res], ignore_index=True)
        # 清洗数据，保存结果
        res_df.drop_duplicates(subset=['选股日期', '交易日期', '股票代码'], keep='last', inplace=True)
        res_df.sort_values(by=['选股日期', '交易日期', '选股排名'], inplace=True)
        res_df.to_csv(save_path, encoding='gbk', index=False)
    else:
        # 不保存文件的话，也要读一下文件有没有，只要没有就创建（否则实盘配置的时候不方便）
        if not os.path.exists(save_path):
            res_df = pd.DataFrame(columns=['选股日期', '交易日期', '股票代码', '股票名称', '选股排名'])
            res_df.to_csv(save_path, encoding='gbk', index=False)


def _factors_linear_regression(data, factor, neutralize_list, industry=None):
    """
    使用线性回归对目标因子进行中性化处理，此方法外部不可直接调用。
    :param data: 股票数据
    :param factor: 目标因子
    :param neutralize_list:中性化处理变量list
    :param industry: 行业字段名称，默认为None
    :return: 中性化之后的数据
    """

    train_col = []
    train_col += neutralize_list

    lrm = LinearRegression(fit_intercept=True)  # 创建线性回归模型
    if industry:  # 如果需要对行业进行中性化，将行业的列名加入到neutralize_list中
        # 获取一下当周期有什么行业，申万一级行业发生过拆分，所以需要考虑
        ind_list = list(data[industry].unique())
        ind_list = ['所属行业_' + ind for ind in ind_list]

        industry_cols = [col for col in data.columns if '所属行业' in col]
        for col in industry_cols:
            if col not in train_col:
                if col in ind_list:
                    train_col.append(col)
    train = data[train_col].copy()  # 输入变量
    label = data[[factor]].copy()  # 预测变量
    lrm.fit(train, label)  # 线性拟合
    predict = lrm.predict(train)  # 输入变量进行预测
    data[factor + '_中性'] = label.values - predict  # 计算残差
    return data


def factor_neutralization(data, factor, neutralize_list, industry=None):
    """
    使用线性回归对目标因子进行中性化处理，此方法可以被外部调用。
    :param data: 股票数据
    :param factor: 目标因子
    :param neutralize_list:中性化处理变量list
    :param industry: 行业字段名称，默认为None
    :return: 中性化之后的数据
    """
    df = data.copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[factor] + neutralize_list, how='any')
    if industry:  # 果需要对行业进行中性化，先构建行业哑变量
        # 剔除中性化所涉及的字段中，包含inf、-inf、nan的部分
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[industry], how='any')
        # 对行业进行哑变量处理
        ind = df[industry]
        ind = pd.get_dummies(ind, columns=[industry], prefix='所属行业',
                             prefix_sep="_", dummy_na=False, drop_first=False)
        """
        drop_first=True会导致某一行业的的哑变量被删除，这样的做的目的是为了消除行业间的多重共线性
        详见：https://www.learndatasci.com/glossary/dummy-variable-trap/

        2023年6月25日起
        不再使用drop_first=True，而指定一个行业直接删除，避免不同的周期删除不同的行业。
        """
        # 删除一个行业，原因如上提到的drop_first
        ind.drop(columns=['所属行业_综合'], inplace=True)
    else:
        ind = pd.DataFrame()
    df = pd.concat([df, ind], axis=1)
    warnings.filterwarnings('ignore')
    df = df.groupby(['交易日期']).apply(_factors_linear_regression, factor=factor,
                                    neutralize_list=neutralize_list, industry=industry)

    df.reset_index(drop=True, inplace=True)
    df.sort_values(by=['交易日期', '股票代码'], inplace=True)
    return df


def merge_offset(equity_list, index_data):
    """
    合并所有offset的策略资金曲线
    :param equity_list: 各offset的资金曲线list
    :param index_data: 指数
    :return: equity_df, equity_df_not_timing：合并完的资金曲线数据,合并完的未择时资金曲线
    """
    # 合并equity_list中所有资金曲线，填充空值，因为不同offset的起始结束日不同，所以肯定有空值
    _equity_df = pd.concat(equity_list, axis=1, join='outer')
    _equity_df.fillna(method='ffill', inplace=True)
    _equity_df.fillna(value=1, inplace=True)

    # 通过最大最小的时间，从index取出需要画图的这段，算完banchmark
    equity_df = index_data[
        (index_data['交易日期'] >= _equity_df.index.min()) & (index_data['交易日期'] <= _equity_df.index.max())].copy()
    equity_df.set_index('交易日期', inplace=True)
    equity_df['benchmark'] = (equity_df['指数涨跌幅'] + 1).cumprod()
    # 合并资金曲线，通过遍历择时和不择时区分两个
    equity_col = _equity_df.columns.unique().to_list()
    for each_col in equity_col:
        equity_df[each_col] = _equity_df[[each_col]].mean(axis=1)
    # 把交易日期变回非index的列
    equity_df.reset_index(drop=False, inplace=True)
    # 资金曲线反推的时候，需要前面加一行，否则第一个涨跌幅算不出
    equity_df = pd.concat([pd.DataFrame([{'equity_curve': 1}]), equity_df], ignore_index=True)
    equity_df['涨跌幅'] = equity_df['equity_curve'] / equity_df['equity_curve'].shift() - 1
    equity_df.drop([0], axis=0, inplace=True)
    if len(equity_col) > 1:
        # 带择时时，需要多算一遍
        equity_df_not_timing = equity_df[['交易日期', '指数涨跌幅', 'benchmark', 'equity_curve_not_timing']].copy()
        equity_df_not_timing.rename(columns={'equity_curve_not_timing': 'equity_curve'}, inplace=True)
        equity_df_not_timing = pd.concat([pd.DataFrame([{'equity_curve': 1}]), equity_df_not_timing], ignore_index=True)
        equity_df_not_timing['涨跌幅'] = equity_df_not_timing['equity_curve'] / equity_df_not_timing[
            'equity_curve'].shift() - 1
        equity_df_not_timing.drop([0], axis=0, inplace=True)
        equity_df = equity_df[['交易日期', '涨跌幅', 'equity_curve', '指数涨跌幅', 'benchmark']]
        equity_df_not_timing = equity_df_not_timing[['交易日期', '涨跌幅', 'equity_curve', '指数涨跌幅', 'benchmark']]
        return equity_df, equity_df_not_timing

    else:
        # 不带择时
        equity_df = equity_df[['交易日期', '涨跌幅', 'equity_curve', '指数涨跌幅', 'benchmark']]
        return equity_df, pd.DataFrame()


def judge_current_period(period_offset, index_data, period_and_offset_df):
    """
    判断指数文件的最后一行日期是否命中周期offset选股日（选股日定义：持仓周期最后一日，尾盘需要卖出持仓，次日早盘买入新的标的）

    :param period_offset: offset周期
    :param index_data: 指数数据
    :param period_and_offset_df: period_offset的dataframe
    :return: True选股日，False非选股日
    """
    index_lastday = index_data['交易日期'].iloc[-1]  # 最新交易日，从指数数据获取
    # 把周期数据转为选股日标记数据
    period_and_offset_df[f'{period_offset}_判断'] = period_and_offset_df[period_offset].abs().diff().shift(-1)
    """
             交易日期  W_0  W_1  W_2  W_0_判断
        0  2005-01-05  0.0  0.0  0.0     0.0
        1  2005-01-06  0.0  0.0  0.0     0.0
        2  2005-01-07  0.0  0.0  0.0     1.0
        3  2005-01-10  1.0  0.0  0.0     0.0
        4  2005-01-11  1.0  1.0  0.0     0.0
        5  2005-01-12  1.0  1.0  1.0     0.0
        6  2005-01-13  1.0  1.0  1.0     0.0
        7  2005-01-14  1.0  1.0  1.0     1.0
        8  2005-01-17  2.0  1.0  1.0     0.0
    """
    # 如果输入的period_offset参数到了换仓的时候，则为选股日，返回True，否则返回False
    if period_and_offset_df.loc[period_and_offset_df['交易日期'] == index_lastday, f'{period_offset}_判断'].iloc[-1] == 1:
        # 选股日
        return True
    else:
        # 非选股日
        return False


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


def merge_timing_data(rtn, rtn_not_timing, year_return, year_return_not_timing, month_return, month_return_not_timing):
    """
    合并带择时后的信息，用于统一print

    :param rtn:择时收益
    :param rtn_not_timing:未择时收益
    :param year_return:年化择时收益
    :param year_return_not_timing:年化未择时收益
    :param month_return: 月度择时收益
    :param month_return_not_timing: 月度未择时收益
    """
    #
    rtn.rename(columns={0: '带择时'}, inplace=True)
    rtn_not_timing.rename(columns={0: '原策略'}, inplace=True)
    rtn = pd.concat([rtn_not_timing, rtn], axis=1)
    year_return = pd.merge(left=year_return_not_timing, right=year_return[['涨跌幅', '超额收益']],
                           left_index=True, right_index=True, how='outer', suffixes=('', '_(带择时)'))
    year_return = year_return[['涨跌幅', '涨跌幅_(带择时)', '指数涨跌幅', '超额收益', '超额收益_(带择时)']]
    month_return = pd.merge(left=month_return_not_timing, right=month_return[['涨跌幅', '超额收益']],
                            left_index=True, right_index=True, how='outer', suffixes=('', '_(带择时)'))
    month_return = month_return[['涨跌幅', '涨跌幅_(带择时)', '指数涨跌幅', '超额收益', '超额收益_(带择时)']]
    return rtn, year_return, month_return


def check_factor_change(factor_path, md5_path):
    """
    检查因子md5是否有变化

    :param factor_path: program.因子的路径
    :param md5_path: 因子MD5记录.csv的路径
    """
    # 获取所有的因子文件信息
    file_list = get_file_in_folder(factor_path, '.py', filters=['__init__'])
    new_md5 = pd.DataFrame()
    for file in file_list:  # 填充new_md5，获取因子名称和MD5码

        # 创建一个md5对象
        md5_hash = hashlib.md5()
        # 创建因子计算py的路径
        file_path = os.path.join(factor_path, file)
        # 打开文件并以二进制模式读取内容
        with open(file_path, "rb") as f:
            # 按块读取文件内容，并不断更新md5对象
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)

        # 获取最终的MD5值（16进制表示）
        file_md5 = md5_hash.hexdigest()

        # 插入的索引
        inx = 0 if pd.isnull(new_md5.index.max()) else new_md5.index.max() + 1
        new_md5.loc[inx, '因子名称'] = file
        new_md5.loc[inx, 'MD5码'] = file_md5

    # 如果md5文件的位置不存在
    if not os.path.exists(md5_path):
        print('因子校验文件不存在，如有因子数据建议全部删除，重新生成\n如果确认无误，可以无视提示，继续执行数据整理任务。')
        new_md5.to_csv(md5_path, encoding='gbk', index=False)
        return

    # 读入历史的校验文件
    old_md5 = pd.read_csv(md5_path, encoding='gbk')
    # 对比新旧数据的差异
    compare_df = pd.merge(new_md5, old_md5, 'left', '因子名称', suffixes=('_new', '_old'))
    compare_df['md5差异'] = compare_df['MD5码_new'] == compare_df['MD5码_old']

    str_info = ''
    # 寻找差值
    diff_df = compare_df[(compare_df['md5差异'] == False) & (compare_df['MD5码_old'].notnull())]
    # 如果存在不一样的md5
    if not diff_df.empty:
        str_info += '\n请注意，以下因子存在变更：\n' + str(
            diff_df['因子名称'].to_list()) + '\n建议删除对应的因子数据，重新生成。'
    # 寻找新的因子
    new_df = compare_df[compare_df['MD5码_old'].isnull()]
    if not new_df.empty:
        str_info += '\n\n请注意，以下因子是新增的：\n' + str(
            new_df['因子名称'].to_list()) + '\n如有因子数据建议全部删除，重新生成'
    if str_info == '':
        print('因子校验正确，继续执行数据整理任务。√')
        return
    str_info += '\n\n如果确认无误，可以无视提示，继续执行数据整理任务，程序先暂停30秒'
    print(str_info)
    # 30秒后自动保存数据，并且保存文件
    time.sleep(3)
    new_md5.to_csv(md5_path, encoding='gbk', index=False)


def use_total(total_path, is_folder=True):
    """
    通过路径判断是否需要保存全部文件
    """
    if not os.path.exists(total_path):  # 如果不存在文件需要全跑
        return True
    elif is_folder and (len(os.listdir(total_path)) == 0):  # 如果文件夹为空也需要全跑
        return True
    else:
        return False


def get_run_info(stg_file, po_df, is_backtest, date, other_factors, floder):
    """
    获取计算前需要的数据，包括因子信息、offset、财务数据等。

    :param stg_file: 策略文件列表
    :param po_df: 所有周期所有offset的df
    :param is_backtest: 是否是回测模式
    :param date: 最新交易日
    :param other_factors: 其他因子，一般是帮轮动策略跑的

    Return:
    fa_info:因子信息
    fa_info存放格式：{
                    因子:{
                         'per_oft':需要计算的周期offset,
                         'cls': program.live_strategies.策略1
                         }
                    因子2:{...}
                  }
    ipt_fin_cols:需要输入的因子列表
    ipt_fin_cols存放格式：['R_np_atoopc@xbx','factor_2', ...]
    opt_fin_cols:需要输出的因子列表
    opt_fin_cols存放格式：['R_np_atoopc@xbx_单季','factor_2', ...]
    load_functions:load函数信息
    load_functions存放格式：{
                           'load_chip_distribution': {
                                                      'func': <function load_chip_distribution at 0x0000029AC375F940>,
                                                      'factors': ['筹码因子']
                                                      }
                           'load_function_2':{'func': ..., 'factors': [...]}
                           }
    """
    stg_list = []
    fa_info = {}
    po_list = []

    # 遍历选股策略下的策略，获取需要的因子以及其对应的周期
    for _file in stg_file:
        cls = __import__('program.%s.%s' % (floder, _file), fromlist=('',))
        stg_list.append(cls)
        # 读取策略下的因子
        for fa in cls.factors.keys():
            # 如果该因子不在fa_info里，添加该因子的period_offset
            if fa not in fa_info.keys():
                fa_info[fa] = {'per_oft': cls.period_offset}
            # 如果有，那么再新加入需要计算的该因子的周期数据
            else:
                fa_info[fa]['per_oft'] = list(set(fa_info[fa]['per_oft'] + cls.period_offset))
            # 检查一下是不是有不规范的列名
            unknown_col = set(cls.period_offset) - set(po_df.columns)
            if len(unknown_col) > 0:
                print(f'{cls.name}策略的{unknown_col}不在period_offset的周期内，请重新配置。')
                exit()
            # 添加当前的周期
            po_list += cls.period_offset

    # 在fa_info中加上需要额外计算的因子
    for ofa in other_factors.keys():
        # 如果其他因子不在要计算的名单里面，最好加一下
        if ofa not in fa_info.keys():
            fa_info[ofa] = {'per_oft': other_factors[ofa]}

    # 去重
    po_list = list(set(po_list))
    # 实盘模式
    if not is_backtest:
        date_inx = po_df[po_df['交易日期'] >= pd.to_datetime(date)].index.min()  # offset文件中最新交易日的索引
        # 找到第1行和第2行不一样的数据，即下一天需要调仓用于实盘的周期offset
        run_po_list = [po for po in po_list if po_df[po].iloc[date_inx] != po_df[po].iloc[date_inx + 1]]
        po_list = run_po_list  # 只考虑实盘要算的offset周期

        # 寻找有需要开仓的策略，其他的策略停掉
        for fa, info in fa_info.items():
            _po = list(set(run_po_list) & set(info['per_oft']))  # 寻找该因子需要计算的并且属于实盘要跑的offset周期
            fa_info[fa]['per_oft'] = _po

        # 只保留需要有周期的因子
        factors = list(fa_info.keys())
        for fa in factors:
            if len(fa_info[fa]['per_oft']) < 1:
                fa_info.pop(fa)

    # 输出一下需要的财务数据字段 & 需要加载的函数
    ipt_fin_cols = []
    opt_fin_cols = []
    load_functions = {}
    # 遍历各个因子获取ipt（输入）和opt（输出）因子
    for fa in fa_info.keys():
        cls = __import__('program.factors.%s' % fa, fromlist=('',))
        # 获取输入因子 ipt
        ipt_fin_cols += cls.ipt_fin_cols
        # 获取输出因子 opt
        opt_fin_cols += cls.opt_fin_cols
        fa_info[fa]['cls'] = cls

        # 看一下是不是有load_开头的函数
        tree = ast.parse(inspect.getsource(cls))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # 如果文件中包含load开头的函数
                if node.name.startswith('load_'):
                    # 如果这个load函数还没被记录下来，则需要先记录一下
                    if node.name not in load_functions.keys():
                        load_functions[node.name] = {'func': getattr(cls, node.name), 'factors': [fa]}
                    else:
                        load_functions[node.name]['factors'].append(fa)
    # 去重
    ipt_fin_cols = list(set(ipt_fin_cols))
    opt_fin_cols = list(set(opt_fin_cols))

    return fa_info, ipt_fin_cols, opt_fin_cols, load_functions, po_list


def _data_process(df, index_data, end_exchange):
    """
    对个股数据进行处理，计算一些必要的数据方便后续回测下单
    :param df: 需要处理的个股数据dataframe
    :param index_data: 指数数据dataframe
    :param end_exchange: 是否为尾盘换仓模式
    """
    # =计算涨跌停价格
    df = cal_zdt_price(df)

    # 转换周期时需要额外处理的字段
    exg_dict = {}  # 重要变量，在将日线数据转换成周期数据时使用。key为需要转换的数据，对应的value为转换的方法，包括first,max,min,last,sum等
    # 和指数合并时需要额外处理的字段
    fill_0_list = ['换手率']  # 在和上证指数合并时使用。
    # 指数成分的列用'N'填充空值
    for col in ['沪深300成分股', '上证50成分股', '中证500成分股', '中证1000成分股', '创业板指成分股']:
        df[col].fillna(value='N', inplace=True)

    # =将股票和上证指数合并，补全停牌的日期，新增数据"是否交易"、"指数涨跌幅"
    df = merge_with_index_data(df, index_data, fill_0_list)
    # =股票退市时间小于指数开始时间，就会出现空值
    if df.empty:
        # 如果出现这种情况，返回空的dataframe用于后续操作
        return pd.DataFrame(), exg_dict

    # =添加不同的价格，方便使用不同的价格买入
    df['均价'] = df['成交额'] / df['成交量']  # 当日成交均价
    # =计算不同价格买入至当天收盘时的涨跌幅
    for col in ['均价', '09:35收盘价', '09:45收盘价', '09:55收盘价']:
        df[f'{col}买入涨跌幅'] = df['收盘价'] / df[col] - 1
        df[f'下日_{col}买入涨跌幅'] = df[f'{col}买入涨跌幅'].shift(-1)  # 获取下日的涨跌幅情况用于回测等分析使用
        exg_dict[f'下日_{col}买入涨跌幅'] = 'last'  # 处理周期时如果需要保存这列数据，需要加入到exg_dict里，last表示保存每段周期最新日期的数据

    # =计算下个交易的相关情况
    df['下日_是否交易'] = df['是否交易'].shift(-1)
    df['下日_一字涨停'] = df['一字涨停'].shift(-1)
    df['下日_开盘涨停'] = df['开盘涨停'].shift(-1)
    df['下日_是否ST'] = df['股票名称'].str.contains('ST').shift(-1)
    df['下日_是否S'] = df['股票名称'].str.contains('S').shift(-1)
    df['下日_是否退市'] = df['股票名称'].str.contains('退').shift(-1)
    df['下日_开盘买入涨跌幅'] = df['开盘买入涨跌幅'].shift(-1)
    # 处理最后一根K线的数据：最后一根K线默认沿用前一日的数据
    state_cols = ['下日_是否交易', '下日_是否ST', '下日_是否S', '下日_是否退市']
    df[state_cols] = df[state_cols].fillna(method='ffill')
    # 非尾盘模式需要将重置下日涨停状态
    if not end_exchange:
        df[['下日_一字涨停', '下日_开盘涨停']] = df[['下日_一字涨停', '下日_开盘涨停']].fillna(value=False)
    return df, exg_dict


def merge_with_hist(increment_df, cut_day, hist_data, is_path=False, waring_info='前后数据不一致'):
    """
    合并历史数据和增量数据
    :param increment_df: dataframe 增量数据
    :param cut_day: 需要保存的起始增量数据的日期
    :param hist_data: dataframe or str 历史数据，如果需要从本地读取，需要is_path=True，同时hist_data给出历史数据路径，否则直接给dataframe进来就行
    :param waring_info: 报警信息
    """

    # 需要删除的增量数据，删除后与存量数据进行拼接
    increment_df = increment_df[increment_df['交易日期'] >= cut_day]
    # 如果hist_data输入的是路径，那么读取这个路径下的本地历史数据
    if is_path:
        hist_data = pd.read_pickle(hist_data)

    # 读取数据的时候顺便对比一下两边的列名是否一致，不一致的话说明改了因子的数据，可能需要重算
    diff = list(set(increment_df.columns) - set(hist_data.columns))
    # 如果存在不一致的列名，输出报警信息
    if len(diff) > 0:
        print(waring_info)
    # 忽略其他报警信息
    warnings.filterwarnings('ignore')
    # 需要删除历史数据的最新的数据，要不然会有问题
    hist_data = hist_data[hist_data['交易日期'] < hist_data['交易日期'].max()]
    # 合并历史数据和增量数据
    total_df = pd.concat([hist_data, increment_df], ignore_index=True)
    # 去掉重复的数据并排序
    total_df = total_df.sort_values(by=['交易日期', '股票代码']).drop_duplicates(subset=['交易日期', '股票代码'],
                                                                         keep='last').reset_index(drop=True)
    return increment_df, total_df


def pre_process(stock_base_path, subset_day, cut_day, df, index_data, po_list, period_offset_df, total_mode,
                end_exchange):
    """
    对数据进行预处理：合并指数、计算下个交易日状态、基础数据resample等
    :param stock_base_path: 基础数据的保存路径
    :param subset_day: 增量计算的开始日期
    :param cut_day: 数据合并的开始日期
    :param df: 读入的股票日线数据
    :param index_data: 指数数据
    :param po_list: 周期列表
    :param period_offset_df: 周期表
    :param total_mode: 是否为强制全量模式
    :param end_exchange: 是否为尾盘换仓模式
    """
    new_total = False  # 是否需要全量计算数据
    # ===判断一下是否有历史数据
    # 如果存在历史数据，只需要计算增量数据然后与历史数据合并
    if os.path.exists(stock_base_path) and (not total_mode):
        # 从本地读取历史数据
        hist_df = pd.read_pickle(stock_base_path)
        # 获取需要计算的增量数据
        increment_df = pd.DataFrame(df[df['交易日期'] >= subset_day])
        # 如果增量数据是空的，直接返回历史数据
        if increment_df.empty:
            return hist_df, new_total
        # 只对增量数据计算涨跌停、合并指数、下日状态等
        increment_df, exg_dict = _data_process(increment_df, index_data, end_exchange)
        # 合并历史数据与增量数据
        increment_df, total_df = merge_with_hist(increment_df, cut_day, hist_df)

    else:  # 如果不存在这个文件，则说明只能计算全量
        # 创建路径
        os.makedirs(os.path.dirname(stock_base_path), exist_ok=True)
        # 全量计算跌停、合并指数、下日状态等
        total_df, exg_dict = _data_process(df, index_data, end_exchange)
        # 全量计算下，new_total=True
        new_total = True

    # 如果得到的全量数据为空，返回空的dataframe
    if total_df.empty:
        return total_df, new_total
    # 保存一下全量数据，这里选择pkl是因为存储和读取比较快，且文件小
    total_df = total_df.sort_values(by='交易日期').reset_index(drop=True)
    total_df.to_pickle(stock_base_path)

    # ===保存基础的周期数据
    # 获取个股代码
    _code = stock_base_path.split('基础数据/')[1]
    # 遍历不同周期
    for po in po_list:
        save_path = os.path.join(stock_base_path.split('日频数据')[0], f'周期数据/{po}/基础数据/')  # 获取该周期下基础数据的路径
        # 新建文件夹，如果已经有文件夹了那就不进行任何操作，也不会报错
        os.makedirs(save_path, exist_ok=True)

        save_path = os.path.join(save_path, _code)
        is_total = use_total(save_path, is_folder=False)  # 判断是否需要存全量数据
        # ===是否存全量数据
        # 如果存全量数据，那么转换全量数据为周期数据后直接保存
        if is_total or new_total or total_mode:
            # 转换成周期数据
            period_df = transfer_to_period_data(total_df, period_offset_df, po, exg_dict)
            period_df.reset_index(drop=True).to_pickle(save_path)
        # 否则，只转换增量数据至周期数据，并和历史数据合并后保存
        else:
            # 转换成周期数据
            period_df = transfer_to_period_data(increment_df, period_offset_df, po, exg_dict)
            # 和历史数据合并
            period_df = merge_with_hist(period_df, cut_day, save_path, True)[1]
            period_df.reset_index(drop=True).to_pickle(save_path)

    return total_df, new_total


def transfer_factor_data(df, po_df, period_offset, extra_agg_dict={}):
    """
    将日线数据转换为相应的周期数据
    :param df:原始数据
    :param po_df:从period_offset.csv载入的数据
    :param period_offset:转换周期
    :param extra_agg_dict:
    :return:
    """
    agg_dict = {'交易日期': 'last', '股票代码': 'last'}
    agg_dict = dict(agg_dict, **extra_agg_dict)
    po_df['_group'] = po_df[period_offset].copy()
    # group为绝对值后的数据，用于对股票数据做groupby
    po_df['group'] = po_df['_group'].abs().copy()
    df = pd.merge(left=df, right=po_df[['交易日期', 'group', '_group']], on='交易日期', how='left')
    # 为了W53（周五买周三卖）这种有空仓日期的周期，把空仓日的涨跌幅设置为0
    df.loc[df['_group'] < 0, '涨跌幅'] = 0

    # ===对个股数据根据周期offset情况，进行groupby后，得到对应的nD/周线/月线数据
    period_df = df.groupby('group').agg(agg_dict)

    # 重置索引
    period_df.reset_index(drop=True, inplace=True)

    return period_df


def load_back_test_data(cls, period_offset, path):
    """
    导入period_offset周期的数据，将prepare_data在该周期下得到的基础数据和因子数据合并并return
    :param cls: 策略
    :param period_offset: offset周期
    :param path: 数据周期保存路径
    """
    # 获取该周期下的基础数据
    base_path = os.path.join(path, f'{period_offset}/基础数据/基础数据.pkl')
    df = pd.read_pickle(base_path)
    factors = cls.factors.keys()
    # 公共列，属于常用的字段，merge时公共字段的列merge不会警告，其他字段的列merge时会发出警告
    common_cols = ['股票代码', '交易日期', '开盘价', '最高价', '最低价', '收盘价', '成交量', '成交额', '流通市值', '总市值', '沪深300成分股', '上证50成分股',
                   '中证500成分股', '中证1000成分股', '中证2000成分股', '创业板指成分股', '新版申万一级行业名称', '新版申万二级行业名称', '新版申万三级行业名称',
                   '09:35收盘价', '09:45收盘价', '09:55收盘价', '复权因子', '收盘价_复权', '开盘价_复权', '最高价_复权', '最低价_复权']
    # 遍历读取因子，合并到基础数据中
    for fa in factors:
        factors_path = os.path.join(path, f'{period_offset}/{fa}/{fa}.pkl')
        fa_df = pd.read_pickle(factors_path)
        # 只获取该策略需要的因子
        if len(cls.factors[fa]) > 0:
            fa_df = fa_df[['交易日期', '股票代码'] + cls.factors[fa]]

        # 对比一下前后列名是否有重复的
        repeat_cols = list(set(df.columns).intersection(set(fa_df.columns)))
        # 要排除掉交易日期和股票代码两列
        repeat_cols = [col for col in repeat_cols if col not in ['股票代码', '交易日期']]
        # 如果还有重复的列
        if len(repeat_cols) > 0:
            for col in repeat_cols:
                if col in common_cols:  # 如果是公共列，则删除
                    fa_df.drop(columns=[col], inplace=True)
                else:
                    print(f'{fa}文件中的{col}列与已经加载的数据重名，程序已经自动退出，请检查因子重名的情况后重新运行')
                    raise Exception(f'{fa}文件中的{col}列与已经加载的数据重名，程序已经自动退出，请检查因子重名的情况后重新运行')
        df = pd.merge(df, fa_df, on=['交易日期', '股票代码'], how='left')
    return df


def trans_code(code):
    """
    更改股票代码的格式，在数字股票代码前加上股票所属交易所的简写
    :param code: 股票代码，如：000001
    """
    start = str(code)[:2]
    if start in ['60', '68']:
        return 'sh' + str(code)
    elif start in ['30', '00']:
        return 'sz' + str(code)
    else:
        return 'bj' + str(code)


def get_stock_data_from_internet(trade_date, count):
    """
    从行情网站获取实时的行情数据
    :param trade_date:      第一个参数，交易日期
    :param count:           第二个参数，股票数量
    :return df:             个股dataframe基础数据
    """
    while True:
        try:
            url = "http://82.push2.eastmoney.com/api/qt/clist/get"
            params = {
                "pn": "1",
                "pz": "50000",
                "po": "1",
                "np": "1",
                "ut": "bd1d9ddb04089700cf9c27f6f7426281",
                "fltt": "2",
                "invt": "2",
                "fid": "f3",
                "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
                "fields": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152",
                "_": "1623833739532",
            }
            # 获取线上的数据
            r = requests.get(url, params=params)
            data_json = r.json()
            # 如果不存在线上数据，等待5s后重试
            if not data_json["data"]["diff"]:
                time.sleep(5)
                continue

            # 将数据转换成我们想要的格式
            temp_df = pd.DataFrame(data_json["data"]["diff"])
            temp_df.columns = ['_', '收盘价', '涨跌幅', '涨跌额', '成交量', '成交额', '振幅', '换手率', '市盈率-动态', '量比', '5分钟涨跌', '股票代码',
                               '_', '股票名称', '最高价', '最低价', '开盘价', '前收盘价', '总市值', '流通市值', '涨速', '市净率', '60日涨跌幅',
                               '年初至今涨跌幅', '-', '-', '-', '-', '-', '-', '-', ]
            # 添加交易日期的列
            temp_df['交易日期'] = trade_date
            temp_df = temp_df[['股票代码', '股票名称', '交易日期', '开盘价', '最高价', '最低价', '收盘价', '前收盘价', '成交量', '成交额', '流通市值', '总市值']]

            # 更改股票代码的格式
            temp_df['股票代码'] = temp_df['股票代码'].apply(trans_code)
            # 超过95%的股票拿到数据了就可以退出了
            stop = len(temp_df) / count > 0.95
            # 剔除不交易的股票
            temp_df = temp_df[temp_df['成交额'] != '-']
            temp_df['成交量'] = temp_df['成交量'].apply(lambda x: 100 * x)
            # 跳出循环
            if stop:
                break
        except:
            print('获取线上数据出错，等待重试')
        # 没抓到数据，或者抓到的数据数量不够都等待5s之后重新抓一下
        time.sleep(5)

    return temp_df


def update_stock_index(index, save_path):
    """
    更新指数数据
    :return:
    """

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                             'Chrome/97.0.4692.71 Safari/537.36 Edg/97.0.1072.62'}
    print(index)
    url = 'https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get'
    start_time = '1900-01-01'
    end_time = ''
    df_list = []  # 创建空列表用于存放线上获取的数据
    while True:
        params = {
            '_var': 'kline_dayqfq',
            'param': f'{index},day,{start_time},{end_time},2000,qfq',
            'r': f'0.{random.randint(10 ** 15, (10 ** 16) - 1)}',
        }
        res = requests.get(url, params=params, headers=headers)
        res_json = json.loads(re.findall('kline_dayqfq=(.*)', res.text)[0])
        if res_json['code'] == 0:
            _df = pd.DataFrame(res_json['data'][index]['day'])
            df_list.append(_df)
            if _df.shape[0] <= 1:
                break
            end_time = _df.iloc[0][0]
        time.sleep(2)
    # 合并获取的多个数据
    df = pd.concat(df_list, ignore_index=True)
    # ===对数据进行整理
    rename_dict = {0: 'candle_end_time', 1: 'open', 2: 'close', 3: 'high', 4: 'low', 5: 'amount', 6: 'info'}
    # 其中amount单位是手，说明数据不够精确
    df.rename(columns=rename_dict, inplace=True)
    # 对列进行重命名，改成正确的字段
    df['candle_end_time'] = pd.to_datetime(df['candle_end_time'])
    df.drop_duplicates('candle_end_time', inplace=True)  # 去重
    df.sort_values('candle_end_time', inplace=True)  # 按照时间进行排序
    df['candle_end_time'] = df['candle_end_time'].dt.strftime('%Y-%m-%d')  # 转换时间格式为年月日形式
    # 如果没有获取到info字段，添加一个空列到df里，保证列格式一致
    if 'info' not in df:
        df['info'] = None
    # 保存数据
    df.to_csv(save_path, index=False, encoding='gbk')


def list_to_str(ipt_list):
    """
    将list格式转换成字符串格式，仅支持可以用字符串表达的list
    :param ipt_list: 输入的list
    :return:
    """
    res_str = ''
    for item in ipt_list:
        res_str += str(item) + '+'
    res_str = res_str[:-1]
    return res_str


def str_to_list(ipt_str, item_type='str'):
    """
    将输入的字符串格式转换会list格式
    :param ipt_str: 输入的字符串
    :param item_type: list中每个元素的类别
    :return:
    """
    res_list = ipt_str.split('+')
    if item_type != 'str':
        for i in range(0, len(res_list)):
            res_list[i] = eval('%s(%s)' % (item_type, res_list[i]))
    return res_list


def save_back_test_result(stg, per_oft, param, rtn, years_ret, path):
    """
    保存回测结果
    :param stg:
    :param per_oft:
    :param param:
    :param rtn:
    :param years_ret:
    :param path:
    :return:
    """
    save_path = os.path.join(path, 'data/回测结果/遍历结果.csv')
    res_df = pd.DataFrame()
    res_df.loc[0, '策略名称'] = stg.name
    res_df.loc[0, '周期&offset'] = per_oft
    res_df.loc[0, '策略参数'] = list_to_str(param)
    res_df.loc[0, '选股数量'] = stg.select_count

    # 把回测指标加到要保存的数据中
    col = '带择时' if '带择时' in rtn.columns else 0
    for i in rtn.index:
        res_df.loc[0, i] = rtn.loc[i, col]
    years = years_ret.copy()
    # 保存历年收益
    years_col = '涨跌幅_(带择时)' if '涨跌幅_(带择时)' in years.columns else '涨跌幅'
    # 有数据的地方开始计算
    years['累计涨跌'] = years[years_col].apply(lambda x: float(x.replace('%', '')))
    years['累计涨跌'] = years['累计涨跌'].cumsum()
    years = years[years['累计涨跌'] != 0]
    # 删除累计涨跌数据

    year_range = str(years.index.min().year) + '_' + str(years.index.max().year)
    year_rtn_info = list_to_str(years[years_col].to_list())
    res_df.loc[0, '年份区间'] = year_range
    res_df.loc[0, '历年收益'] = year_rtn_info

    # 保存文件
    if os.path.exists(save_path):
        res_df.to_csv(save_path, encoding='gbk', index=False, header=False, mode='a')
    else:
        res_df.to_csv(save_path, encoding='gbk', index=False)

    return


def _cal_stock_weight_of_each_period(group, date_df):
    """
    计算每个个股每个周期的权重
    :param group:
    :param date_df:
    :return:
    """
    # 将个股数据与周期数据合并
    group = pd.merge(date_df, group, 'left', '交易日期')
    # 填充空数据
    group['股票代码'].fillna(value=group[group['股票代码'].notnull()]['股票代码'].iloc[0], inplace=True)
    group['占比'].fillna(value=0, inplace=True)

    # 获取上周期占比和下周期占比
    group['上周期占比'] = group['占比'].shift(1).fillna(value=0)
    group['下周期占比'] = group['占比'].shift(-1).fillna(value=0)

    # 计算开仓比例
    group['开仓比例'] = group['占比'] - group['上周期占比']
    group['开仓比例'] = group['开仓比例'].apply(lambda x: x if x > 0 else 0)

    # 计算平仓比例
    group['平仓比例'] = group['占比'] - group['下周期占比']
    group['平仓比例'] = group['平仓比例'].apply(lambda x: x if x > 0 else 0)

    return group


def _cal_next_period_pct_change(row, Cfg):
    """
    计算下个周期扣除手续费后的涨跌幅
    :param row:
    :param Cfg:
    :return:
    """
    # 扣除买入手续费
    row['选股下周期每天资金曲线'] = row['选股下周期每天资金曲线'] * (1 - Cfg.c_rate * row['开仓比例'])  # 计算有不精准的地方
    # 扣除卖出手续费
    row['选股下周期每天资金曲线'] = list(row['选股下周期每天资金曲线'][:-1]) + [
        row['选股下周期每天资金曲线'][-1] * (1 - row['平仓比例'] * (Cfg.c_rate + Cfg.t_rate))]

    return row['选股下周期每天资金曲线']


def cal_fee_rate(df, Cfg):
    """
    计算手续费，
    :param select_stock:
    :param Cfg:
    :return:
    """

    # ===挑选出选中股票
    df['股票代码'] += ' '
    df['股票名称'] += ' '
    group = df.groupby('交易日期')
    select_stock = pd.DataFrame()
    # ===统计一些数据用于后续计算
    select_stock['股票数量'] = group['股票名称'].size()
    select_stock['买入股票代码'] = group['股票代码'].sum()
    select_stock['买入股票名称'] = group['股票名称'].sum()

    # =====计算资金曲线
    # 计算下周期每天的资金曲线
    select_stock['选股下周期每天资金曲线'] = group['下周期每天涨跌幅'].apply(
        lambda x: np.cumprod(np.array(list(x)) + 1, axis=1).mean(axis=0))

    # 非收盘模式的换仓，会在上一个交易日全部卖掉，本交易日全部买回，相对简单
    if Cfg.buy_method != '收盘':
        # 扣除买入手续费
        select_stock['选股下周期每天资金曲线'] = select_stock['选股下周期每天资金曲线'] * (1 - Cfg.c_rate)  # 计算有不精准的地方
        # 扣除卖出手续费、印花税。最后一天的资金曲线值，扣除印花税、手续费
        select_stock['选股下周期每天资金曲线'] = select_stock['选股下周期每天资金曲线'].apply(
            lambda x: list(x[:-1]) + [x[-1] * (1 - Cfg.c_rate - Cfg.t_rate)])

    # 收盘模式换仓，对比要卖出和要买入的股票相同的部分，针对相同的部分进行买卖交易
    else:
        # 针对计算各个股票的占比
        stocks = df[['交易日期', '股票代码']].sort_values(['交易日期', '股票代码']).reset_index(drop=True)
        stocks['占比'] = 1 / stocks.groupby('交易日期').transform('count')

        # 计算日期序列
        date_df = pd.DataFrame(stocks['交易日期'].unique(), columns=['交易日期'])

        # 计算每个个股每个周期的权重
        stocks = stocks.groupby('股票代码').apply(lambda g: _cal_stock_weight_of_each_period(g, date_df))
        stocks.reset_index(inplace=True, drop=True)

        # 计算当前周期的开仓比例和平仓比例
        select_stock['开仓比例'] = stocks.groupby('交易日期')['开仓比例'].sum()
        select_stock['平仓比例'] = stocks.groupby('交易日期')['平仓比例'].sum()

        # 扣除交易的手续费
        select_stock['选股下周期每天资金曲线'] = select_stock.apply(lambda row: _cal_next_period_pct_change(row, Cfg), axis=1)

    return select_stock





