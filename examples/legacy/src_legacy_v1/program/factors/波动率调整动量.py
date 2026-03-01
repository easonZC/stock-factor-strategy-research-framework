"""
Factor definition module: 波动率调整动量.
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

def cal_factors(data, fin_data, fin_raw_data, exg_dict):
    
    try:
        # 参数设置
        MOMENTUM_WINDOW = 20
        VOLATILITY_WINDOW = 10
        MIN_PERIODS = 5
        
        # 确保数据排序
        data = data.sort_values(['股票代码', '交易日期'])
        
        # 计算日收益率
        data['日收益率'] = data.groupby('股票代码')['收盘价'].pct_change()
        
        # 计算动量
        data['动量'] = data.groupby('股票代码')['收盘价'].transform(
            lambda s: s.pct_change(periods=MOMENTUM_WINDOW)
        )
        
        # 计算波动率
        data['波动率'] = data.groupby('股票代码')['日收益率'].transform(
            lambda s: s.rolling(VOLATILITY_WINDOW, min_periods=MIN_PERIODS).std()
        )
        
        # 计算波动率调整动量
        data['优化波动率动量'] = data['动量'] / data['波动率'].replace(0, np.nan)
        
        # 处理极端值
        data['优化波动率动量'] = np.clip(data['优化波动率动量'], -5, 5)
        
        # 添加动态止损
        STOP_LOSS_MULTIPLIER = 0.02
        data['动态止损位'] = data.groupby('股票代码')['收盘价'].transform(
            lambda s: s.rolling(5).max() * (1 - STOP_LOSS_MULTIPLIER)
        )
        
        exg_dict['优化波动率动量'] = 'last'
        exg_dict['动态止损位'] = 'last'
        
        print("波动率调整动量因子计算成功！")
        
    except Exception as e:
        print(f"波动率调整动量因子计算失败: {e}")
        if '优化波动率动量' not in data.columns:
            data['优化波动率动量'] = np.nan
        if '动态止损位' not in data.columns:
            data['动态止损位'] = np.nan
    
    return data, exg_dict


