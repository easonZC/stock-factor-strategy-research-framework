"""
Factor definition module: 波动率压缩.
"""

name = file_name = __file__.replace('\\', '/').split('/')[-1].replace('.py', '')  # 当前文件的名字

ipt_fin_cols = []  # 输入的财务字段，财务数据上原始的

opt_fin_cols = []  # 输出的财务字段，需要保留的


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
    # 计算波动率序列
    import numpy as np
    window = 20

    data['returns'] = data['收盘价'].pct_change()
    volatility = data['returns'].rolling(5).std()
    
    # 计算压缩特征
    vol_mean = volatility.rolling(window, min_periods = 1).mean()
    vol_std = volatility.rolling(window, min_periods = 1).std()
    compression = (vol_mean - volatility) / (vol_mean + 1e-9)  # 相对偏离度
    
    # 波动率通道收窄特征
    boll_width = data['收盘价'].rolling(window, min_periods = 1).std() / data['收盘价'].rolling(window, min_periods = 1).mean()
    width_ratio = boll_width / boll_width.rolling(60).mean()
    data['波动率压缩'] = 0.7*compression + 0.3*(1-width_ratio)

    # 添加到exg_dict中以便保存
    exg_dict['波动率压缩'] = 'last'

    return data, exg_dict

def after_resample(data):
    '''
    数据降采样之后的处理流程，非必要
    :param data: 传入的数据
    :return:
    '''
    return data


def cal_cross_factor(data):
    '''
    截面处理数据
    data: 全部的股票数据
    '''

    return data


