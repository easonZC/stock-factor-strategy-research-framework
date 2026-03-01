"""
Factor definition module: ATR.
"""

name = file_name = __file__.replace('\\', '/').split('/')[-1].replace('.py', '')

ipt_fin_cols = []  # 输入的财务字段
opt_fin_cols = []  # 输出的财务字段

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def special_data():
    '''
    初始化数据结构 - 存储前一日收盘价
    '''
    return {
        'prev_close': {}  # 股票代码: 前一日收盘价
    }

def after_resample(data):
    return data

def cal_cross_factor(data):
    return data

def cal_factors(data, fin_data, fin_raw_data, exg_dict):
    """
    计算ATR因子和动态止损位
    完全适配图片描述的逻辑
    """
    print("===== 开始计算ATR因子 =====")
    
    # 参数设置
    ATR_PERIOD = 14  # 标准ATR计算周期
    K_MULTIPLIER = 2  # 止损倍数（图片中蓝色线）
    
    # 确保数据按股票代码和时间排序
    data = data.sort_values(['股票代码', '交易日期'])
    
    # 初始化特殊数据结构
    if not hasattr(data, 'special_data'):
        data.special_data = special_data()
    
    # 计算真实波幅(TR)
    print("计算真实波幅(TR)...")
    data['prev_close'] = data.groupby('股票代码')['收盘价'].shift(1)
    
    # 计算三种波幅
    data['tr1'] = data['最高价'] - data['最低价']  # 当日高低差
    data['tr2'] = np.abs(data['最高价'] - data['prev_close'])  # 当日最高与前收差
    data['tr3'] = np.abs(data['最低价'] - data['prev_close'])  # 当日最低与前收差
    
    # 真实波幅 = max(tr1, tr2, tr3)
    data['TR'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # 计算ATR（14日平均真实波幅）
    print("计算ATR...")
    data['ATR'] = data.groupby('股票代码')['TR'].transform(
        lambda s: s.rolling(window=ATR_PERIOD, min_periods=1).mean()
    )
    
    # 计算动态止损位（图片中的蓝色阶梯线）
    print("计算动态止损位...")
    data['dynamic_stop'] = data['收盘价'] - K_MULTIPLIER * data['ATR']
    
    # 注册因子
    exg_dict['ATR'] = 'last'
    exg_dict['dynamic_stop'] = 'last'
    
    print("===== ATR因子计算完成 =====")
    return data, exg_dict


