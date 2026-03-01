"""
Factor definition module: 净利润率.
"""

name = file_name = __file__.replace('\\', '/').split('/')[-1].replace('.py', '')  # 当前文件的名字

ipt_fin_cols = ['R_revenue@xbx','R_np@xbx']  # 输入的财务字段，财务数据上原始的

opt_fin_cols = ['R_revenue@xbx_ttm','R_np@xbx_ttm']  # 输出的财务字段，需要保留的


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
    # 净利润率 = 净利润 / 营业收入
    # R_np@xbx_ttm：利润表的净利润ttm
    # R_revenue@xbx_ttm:利润表的营业收入ttm
    data['净利润率'] = data['R_np@xbx_ttm'] / data['R_revenue@xbx_ttm']
    exg_dict['净利润率'] = 'last'

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



