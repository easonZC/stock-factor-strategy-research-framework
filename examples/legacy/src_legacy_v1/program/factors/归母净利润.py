"""
Factor definition module: 归母净利润.
"""

import program.runtime_utils as Fun

name = file_name = __file__.replace('\\', '/').split('/')[-1].replace('.py', '')  # 当前文件的名字

ipt_fin_cols = ['R_np_atoopc@xbx']  # 输入的财务字段，财务数据上原始的

opt_fin_cols = ['R_np_atoopc@xbx_单季']  # 输出的财务字段，需要保留的


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
    data['单季度归母净利润'] = data['R_np_atoopc@xbx_单季']
    exg_dict['单季度归母净利润'] = 'last'
    exg_dict['总市值'] = 'last'

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
    data = Fun.factor_neutralization(data, factor='单季度归母净利润', neutralize_list=['总市值'])
    data.drop(columns=['总市值'], inplace=True)
    return data




