"""
Factor definition module: 动量.
"""

name = file_name = __file__.replace('\\', '/').split('/')[-1].replace('.py', '')

ipt_fin_cols = []  # 输入的财务字段
opt_fin_cols = []  # 输出的财务字段

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def special_data():
    return

def after_resample(data):
    return data

def cal_cross_factor(data):
    return data

def calculate_turnover(data):
    # 解决市值单位问题
    sample_data = data.sample(min(1000, len(data)))
    volume_to_marketcap = sample_data['成交量'] / sample_data['流通市值']
    if volume_to_marketcap.median() > 100:
        data['流通市值_元'] = data['流通市值'] * 10000  # 万元转元
    else:
        data['流通市值_元'] = data['流通市值']
    
    # 计算换手率
    data['流通股本'] = data['流通市值_元'] / data['收盘价']
    data['换手率'] = data['成交量'] / data['流通股本']
    data['换手率'] = np.clip(data['换手率'], 0, 1)
    
    return data

def simple_momentum(data, period_months):

    trading_days = int(period_months * 20)
    data[f'return_{period_months}m'] = data.groupby('股票代码')['收盘价'].transform(
        lambda x: x.pct_change(periods=trading_days)
    )
    return data

def weighted_momentum(data, period_months):
    """
    换手率加权动量因子
    """
    trading_days = int(period_months * 20)
    factor_name = f'wgt_return_{period_months}m'
    data[factor_name] = np.nan
    
    for symbol, group in data.groupby('股票代码'):
        group = group.sort_values('交易日期')
        if len(group) < trading_days + 1:
            continue
            
        result = []
        for i in range(trading_days, len(group)):
            window_data = group.iloc[i-trading_days:i+1].copy()
            window_data['daily_return'] = window_data['收盘价'].pct_change()
            window_data = window_data.iloc[1:]
            
            total_weight = window_data['换手率'].sum()
            if total_weight > 0:
                weighted_return = (window_data['daily_return'] * window_data['换手率']).sum() / total_weight
                result.append(weighted_return)
            else:
                result.append(np.nan)
        
        data.loc[group.index[trading_days:], factor_name] = result
    
    return data

def exp_weighted_momentum(data, period_months):
    """
    指数衰减加权动量因子
    """
    trading_days = int(period_months * 20)
    factor_name = f'exp_wgt_return_{period_months}m'
    data[factor_name] = np.nan
    
    for symbol, group in data.groupby('股票代码'):
        group = group.sort_values('交易日期')
        if len(group) < trading_days + 1:
            continue
            
        result = []
        for i in range(trading_days, len(group)):
            window_data = group.iloc[i-trading_days:i+1].copy()
            window_data['daily_return'] = window_data['收盘价'].pct_change()
            window_data = window_data.iloc[1:]
            
            # 计算指数衰减权重
            window_data['days_ago'] = range(len(window_data)-1, -1, -1)
            window_data['exp_weight'] = np.exp(-window_data['days_ago'] * (4/trading_days))
            window_data['combined_weight'] = window_data['换手率'] * window_data['exp_weight']
            total_weight = window_data['combined_weight'].sum()
            
            if total_weight > 0:
                weighted_return = (window_data['daily_return'] * window_data['combined_weight']).sum() / total_weight
                result.append(weighted_return)
            else:
                result.append(np.nan)
        
        data.loc[group.index[trading_days:], factor_name] = result
    
    return data

def after_resample(data):
    return data

def cal_cross_factor(data):
    return data

def cal_factors(data, fin_data, fin_raw_data, exg_dict):
    
    try:
        # 步骤1：计算换手率
        data = calculate_turnover(data)
        
        # 步骤2：计算简单动量因子
        for months in [1, 3, 6, 12]:
            data = simple_momentum(data, months)
            exg_dict[f'return_{months}m'] = 'last'
        
        # 步骤3：计算换手率加权动量因子（修复）
        for months in [1, 3, 6, 12]:
            data = weighted_momentum(data, months)
            exg_dict[f'wgt_return_{months}m'] = 'last'
        
        # 步骤4：计算指数衰减加权动量因子（修复）
        for months in [1, 3, 6, 12]:
            data = exp_weighted_momentum(data, months)
            exg_dict[f'exp_wgt_return_{months}m'] = 'last'
        return data, exg_dict
    except Exception as e:
        print(f"动量因子计算失败: {e}")
        import traceback
        traceback.print_exc()  # 打印详细错误信息
    
    print("动量因子计算完成！")
    return data, exg_dict


