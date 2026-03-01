"""
Factor definition module: RVI.
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
    import numpy as np
    
    # 1. 计算日收益率 - 使用pandas的pct_change()方法
    data['returns'] = data['收盘价'].pct_change()
    
    # 2. 计算波动率序列
    std_dev = data['returns'].rolling(window=5, min_periods=1).std()
    
    # 3. 分离上涨/下跌波动分量
    up_vol = std_dev.where(data['returns'] > 0, 0)
    down_vol = std_dev.where(data['returns'] < 0, 0)
    
    # 4. 计算RVI
    up_mean = up_vol.ewm(span=5).mean()
    down_mean = down_vol.ewm(span=5).mean()
    
    # 避免除零错误
    denominator = up_mean + down_mean
    data['RVI'] = np.where(denominator != 0, 100 * (up_mean / denominator), 50)
    
    # 将RVI添加到exg_dict中以便保存
    exg_dict['RVI'] = 'last'
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


