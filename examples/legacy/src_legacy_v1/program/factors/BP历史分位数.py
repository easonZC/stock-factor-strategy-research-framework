"""
Factor definition module: BP历史分位数.
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

    # 参数设置
    ROLLING_WINDOW = 252
    MIN_PERIODS = 60 

    data['BP'] = data['净资产'] / data['总市值']
    
    if data['BP'].median() < 0.001:
        print("检测到单位不匹配，进行转换...")
        data['BP'] = data['净资产'] / (data['总市值'] * 10000)
    
    print("计算日频滚动历史分位数...")
    data['hist_rank'] = data.groupby('股票代码')['BP'].transform(
        lambda s: s.rolling(
            window=ROLLING_WINDOW,
            min_periods=MIN_PERIODS
        ).apply(
            lambda x: (x.rank().iloc[-1] - 1) / max(1, len(x.dropna()) - 1)
        )
    )
    
    exg_dict['hist_rank'] = 'last'
    
    return data, exg_dict


