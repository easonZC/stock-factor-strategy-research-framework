"""
Factor definition module: 量稳换手率变化率.
"""

name = file_name = __file__.replace('\\', '/').split('/')[-1].replace('.py', '')

ipt_fin_cols = []  # 输入的财务字段
opt_fin_cols = []  # 输出的财务字段

import pandas as pd
import numpy as np

def special_data():
    '''
    处理策略需要的专属数据，非必要。
    :return:
    '''

    return

def cal_factors(data, fin_data, fin_raw_data, exg_dict):
    '''
    合并数据后计算策略需要的因子，非必要
    :param data:传入的数据
    :param fin_data:财报数据（去除废弃研报)
    :param fin_raw_data:财报数据（未去除废弃研报）
    :param exg_dict:resample规则
    :return:
    '''

    k_window = 20
    x_window = 40
    
    # 计算SCR因子
    data['SCR'] = calculate_scr_factor(data, k_window, x_window)
    exg_dict['SCR'] = 'last'
    
    return data, exg_dict

def calculate_scr_factor(data, k=20, x=40):
    """
    计算量稳换手率变化率(SCR)因子
    1. 计算当月换手率波动率
    2. 计算基准换手率波动率
    3. SCR = (当月波动率/基准波动率) - 1
    4. 市值中性化处理
    """
    # 1. 计算当月换手率波动率（S1）
    data['S1'] = data.groupby('股票代码')['换手率'].transform(
        lambda s: s.rolling(k).std()
    )
    
    # 2. 计算基准换手率波动率（S2）
    data['S2'] = data.groupby('股票代码')['换手率'].transform(
        lambda s: s.rolling(x).std().shift(k)
    )
    
    # 3. 计算SCR = S1/S2 - 1
    data['SCR_raw'] = data['S1'] / data['S2'] - 1
    
    # 4. 市值中性化处理
    return market_cap_neutralization(data, 'SCR_raw')

def market_cap_neutralization(data, factor_col):
    """
    市值中性化处理
    :param data: 包含因子和市值的DataFrame
    :param factor_col: 待中性化的因子列名
    :return: 中性化后的因子Series
    """
    # 按交易日分组进行横截面回归
    neutralized = pd.Series(index=data.index, dtype=float)
    
    for date, group in data.groupby('交易日期'):
        X = np.log(group['总市值']).values.reshape(-1, 1)  # 对数市值
        y = group[factor_col].values
        
        # 线性回归
        if len(y) > 1 and not np.isnan(X).any() and not np.isnan(y).any():
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            residuals = y - model.predict(X)
            neutralized.loc[group.index] = residuals
        else:
            neutralized.loc[group.index] = y
    
    return neutralized

def after_resample(data):
    return data

def cal_cross_factor(data):
    return data




