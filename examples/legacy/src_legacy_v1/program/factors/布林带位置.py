"""
Factor definition module: 布林带位置.
"""

name = file_name = __file__.replace('\\', '/').split('/')[-1].replace('.py', '')

ipt_fin_cols = []
opt_fin_cols = []

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
    """
    """
    try:
        data = data.sort_values(['股票代码', '交易日期'])
        
        window = 20
        data['SMA_20'] = data.groupby('股票代码')['收盘价'].transform(
            lambda s: s.rolling(window).mean()
        )
        data['std_20'] = data.groupby('股票代码')['收盘价'].transform(
            lambda s: s.rolling(window).std()
        )
        
        data['BB_upper'] = data['SMA_20'] + 2 * data['std_20']
        data['BB_lower'] = data['SMA_20'] - 2 * data['std_20']
        
        data['BB_position'] = (data['收盘价'] - data['BB_lower']) / (
            data['BB_upper'] - data['BB_lower']
        )
        
        exg_dict['BB_position'] = 'last'
        
        print("布林带位置因子计算成功！")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        data['BB_position'] = 0.5  
    exg_dict['BB_position'] = 'last'
    return data, exg_dict


