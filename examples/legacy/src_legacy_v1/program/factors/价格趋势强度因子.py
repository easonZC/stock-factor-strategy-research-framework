"""
Factor definition module: 价格趋势强度因子.
"""

name = file_name = __file__.replace('\\', '/').split('/')[-1].replace('.py', '')

ipt_fin_cols = []  # 输入的财务字段
opt_fin_cols = []  # 输出的财务字段

import pandas as pd
import numpy as np

def special_data():
    return

def after_resample(data):
    return data

def cal_cross_factor(data):
    return data

def cal_factors(data, fin_data, fin_raw_data, exg_dict):
    """
    """

    ADX_PERIOD = 14
    
    try:
        data = data.sort_values(['股票代码', '交易日期'])
        
        # 计算真实波幅（TR）
        data['tr1'] = data['最高价'] - data['最低价']
        data['tr2'] = np.abs(data['最高价'] - data['收盘价'].shift(1))
        data['tr3'] = np.abs(data['最低价'] - data['收盘价'].shift(1))
        data['tr'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # 计算方向移动（DM）
        data['up_move'] = data['最高价'] - data['最高价'].shift(1)
        data['down_move'] = data['最低价'].shift(1) - data['最低价']
        data['plus_dm'] = np.where((data['up_move'] > data['down_move']) & (data['up_move'] > 0), data['up_move'], 0)
        data['minus_dm'] = np.where((data['down_move'] > data['up_move']) & (data['down_move'] > 0), data['down_move'], 0)
        
        # 计算平滑TR和DM
        data['smoothed_tr'] = data.groupby('股票代码')['tr'].transform(
            lambda s: s.rolling(ADX_PERIOD).mean()
        )
        data['smoothed_plus_dm'] = data.groupby('股票代码')['plus_dm'].transform(
            lambda s: s.rolling(ADX_PERIOD).mean()
        )
        data['smoothed_minus_dm'] = data.groupby('股票代码')['minus_dm'].transform(
            lambda s: s.rolling(ADX_PERIOD).mean()
        )
        
        # 计算方向指数（DI）
        data['plus_di'] = 100 * data['smoothed_plus_dm'] / data['smoothed_tr']
        data['minus_di'] = 100 * data['smoothed_minus_dm'] / data['smoothed_tr']
        
        # 计算方向差异和总和
        data['di_diff'] = np.abs(data['plus_di'] - data['minus_di'])
        data['di_sum'] = data['plus_di'] + data['minus_di']
        
        # 计算ADX
        data['dx'] = 100 * data['di_diff'] / data['di_sum'].replace(0, np.nan)
        data['ADX'] = data.groupby('股票代码')['dx'].transform(
            lambda s: s.rolling(ADX_PERIOD).mean()
        )

        exg_dict['ADX'] = 'last'
        
        
    except Exception as e:
        print(f"ADX因子计算失败: {e}")
        if 'ADX' not in data.columns:
            data['ADX'] = np.nan
    
    return data, exg_dict


