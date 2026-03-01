"""
Factor definition module: 成交额周月比.
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
    # 成交额周月比 = 近一周成交额平均值 / 近一个月成交额平均值
    data['成交额5'] = data['成交额'].rolling(5, min_periods=1).mean()
    data['成交额20'] = data['成交额'].rolling(20, min_periods=1).mean()
    data['成交额周月比'] = data['成交额5'] / data['成交额20']

    # 只保留成交额周月比小于1的信号
    data.loc[data['成交额周月比'] >= 1, '成交额周月比'] = float('nan')

    exg_dict['成交额周月比'] = 'last'

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



