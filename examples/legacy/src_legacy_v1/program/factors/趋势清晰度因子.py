"""
Factor definition module: 趋势清晰度因子.
"""

name = file_name = __file__.replace('\\', '/').split('/')[-1].replace('.py', '')

ipt_fin_cols = []  # 输入的财务字段
opt_fin_cols = []  # 输出的财务字段

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def special_data():
    '''
    初始化数据结构 - 存储每只股票的历史收盘价序列和时间序列
    '''
    return {
        'close_series': {},     # 股票代码: [历史收盘价]
        'trade_date_series': {}  # 股票代码: [历史交易日期]
    }

def after_resample(data):
    return data

def cal_cross_factor(data):
    # cal_cross_factor不再使用
    return data

def cal_factors(data, fin_data, fin_raw_data, exg_dict):
    """
    完整计算趋势清晰度因子(TC)、传统动量因子(MOM)及趋势清晰动量因子(TM1, TM2)
    """
    print(f"开始计算趋势清晰动量因子，数据形状: {data.shape}")
    
    # 确保数据按股票代码和时间排序
    data = data.sort_values(['股票代码', '交易日期'])
    
    # 初始化特殊数据结构
    if not hasattr(data, 'special_data'):
        data.special_data = special_data()
    
    print("存储每只股票的历史数据序列...")
    # 存储每只股票的收盘价和日期序列
    for symbol, group in data.groupby('股票代码'):
        # 只存储必要的数据以节省内存
        if symbol not in data.special_data['close_series']:
            data.special_data['close_series'][symbol] = group['收盘价'].tolist()
            data.special_data['trade_date_series'][symbol] = group['交易日期'].tolist()
    
    # 计算传统动量因子（MOM_240 - 240日动量）
    print("计算传统动量因子(MOM_240)...")
    data['前收盘价'] = data.groupby('股票代码')['收盘价'].shift(240)
    data['MOM_240'] = data['收盘价'] / data['前收盘价'] - 1
        
    # 计算趋势清晰度因子（TC）
    print("计算趋势清晰度因子(TC)...")
    data['趋势清晰度'] = data.apply(
        lambda row: calculate_tc(
            symbol=row['股票代码'],
            current_index=row.name,
            special_data=data.special_data,
            formation_period=240  # 形成期240交易日
        ), 
        axis=1
        )
        
    # 为计算TM1/TM2准备数据
    print("为计算TM1/TM2准备数据...")
    # 复制一份按日期排序的数据用于横截面计算
    tm_data = data.copy().sort_values('交易日期')
        
        # 按交易日分组计算横截面标准化
    print("进行横截面标准化...")
    for date in tm_data['交易日期'].unique():
        date_mask = tm_data['交易日期'] == date
        date_data = tm_data[date_mask]
            
        # 标准化MOM_240
        if date_data['MOM_240'].std() > 1e-5:
            tm_data.loc[date_mask, 'MOM_prime'] = (
                    (date_data['MOM_240'] - date_data['MOM_240'].mean()) / 
                    date_data['MOM_240'].std()
                )
        else:
            tm_data.loc[date_mask, 'MOM_prime'] = 0
                
        # 标准化趋势清晰度
        if date_data['趋势清晰度'].std() > 1e-5:
            tm_data.loc[date_mask, 'TC_prime'] = (
                    (date_data['趋势清晰度'] - date_data['趋势清晰度'].mean()) / 
                    date_data['趋势清晰度'].std()
                )
        else:
            tm_data.loc[date_mask, 'TC_prime'] = 0
        
    # 计算趋势清晰动量因子1 (TM1) = sign(MOM') * TC'
    print("计算TM1...")
    tm_data['TM1'] = np.sign(tm_data['MOM_prime']) * tm_data['TC_prime']
        
    # 计算趋势清晰动量因子2 (TM2) = -|MOM' - TC'|
    print("计算TM2...")
    tm_data['TM2'] = -np.abs(tm_data['MOM_prime'] - tm_data['TC_prime'])
        
    # 将结果合并回原始数据
    data = pd.merge(
            data, 
            tm_data[['股票代码', '交易日期', 'MOM_prime', 'TC_prime', 'TM1', 'TM2']],
            on=['股票代码', '交易日期'],
            how='left'
        )
    # 注册因子扩展规则
    exg_dict['MOM_240'] = 'last'
    exg_dict['趋势清晰度'] = 'last'
    exg_dict['MOM_prime'] = 'last'
    exg_dict['TC_prime'] = 'last'
    exg_dict['TM1'] = 'last'
    exg_dict['TM2'] = 'last'
    
    print(f"因子计算完成，数据形状: {data.shape}")
    
    return data, exg_dict

def calculate_tc(symbol, current_index, special_data, formation_period=240):
    """
    计算单只股票在特定时点的趋势清晰度因子(TC)
    
    参数说明：
    symbol - 股票代码
    current_index - 当前数据点在原始DataFrame中的索引(行号)
    special_data - 存储历史数据的数据结构
    formation_period - 形成期长度
    """
    # 获取该股票的历史数据
    close_prices = special_data['close_series'].get(symbol, [])
    trade_dates = special_data['trade_date_series'].get(symbol, [])
    
    # 检查数据是否足够
    if len(close_prices) < formation_period or current_index < formation_period:
        return np.nan
    
    # 确定形成期窗口（T-240到T-20）
    start_idx = current_index - formation_period
    end_idx = current_index - 20  # 剔除最近20个交易日
    
    # 检查索引是否有效
    if start_idx < 0 or end_idx < start_idx or end_idx > len(close_prices):
        return np.nan
    
    # 提取形成期数据
    formation_prices = close_prices[start_idx:end_idx]
    formation_dates = trade_dates[start_idx:end_idx]
    
    # 检查最小交易日要求
    if len(formation_prices) < 200:  # 至少200个交易日数据
        return np.nan
    
    # 将日期转换为时间序列数值（计算交易天数差）
    base_date = formation_dates[0]
    date_nums = []
    for d in formation_dates:
        try:
            days_diff = (pd.to_datetime(d) - pd.to_datetime(base_date)).days
            date_nums.append(days_diff)
        except:
            continue
    
    if len(date_nums) < 180:  # 确保足够的数据点
        return np.nan
    
    date_nums = np.array(date_nums).reshape(-1, 1)
    prices = np.array(formation_prices[:len(date_nums)])  # 确保长度一致
    
    # 线性回归计算R方
    model = LinearRegression()
    model.fit(date_nums, prices)
    r_squared = model.score(date_nums, prices)
    return r_squared


