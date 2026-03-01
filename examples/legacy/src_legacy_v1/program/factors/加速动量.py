"""
Factor definition module: 加速动量.
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
        SHORT_WINDOW = 10
        LONG_WINDOW = 20
        STOP_LOSS_MULTIPLIER = 0.02
        
        # 确保数据排序
        data = data.sort_values(['股票代码', '交易日期'])
        
        # 计算短期动量
        data['短期动量'] = data.groupby('股票代码')['收盘价'].transform(
            lambda s: s.pct_change(periods=SHORT_WINDOW)
        )
        
        # 计算长期动量
        data['长期动量'] = data.groupby('股票代码')['收盘价'].transform(
            lambda s: s.pct_change(periods=LONG_WINDOW)
        )
        
        # 计算加速动量
        data['优化加速动量'] = data['短期动量'] - data['长期动量']
        data['优化加速动量'] = np.clip(data['优化加速动量'], -0.5, 0.5)
        
        # 添加动态止损
        data['动态止损位'] = data.groupby('股票代码')['收盘价'].transform(
            lambda s: s.rolling(5).max() * (1 - STOP_LOSS_MULTIPLIER)
        )
        
        # 注册因子
        exg_dict['优化加速动量'] = 'last'
        exg_dict['动态止损位'] = 'last'
        
        print("加速动量因子计算成功！")
        
    except Exception as e:
        print(f"加速动量因子计算失败: {e}")
        # 确保列存在
        if '优化加速动量' not in data.columns:
            data['优化加速动量'] = np.nan
        if '动态止损位' not in data.columns:
            data['动态止损位'] = np.nan
    
    return data, exg_dict


