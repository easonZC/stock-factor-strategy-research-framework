"""
Factor definition module: ROE.
"""

name = file_name = __file__.replace('\\', '/').split('/')[-1].replace('.py', '')  # 当前文件的名字

ipt_fin_cols = ['R_np_atoopc@xbx','B_total_equity_atoopc@xbx']  # 输入的财务字段，财务数据上原始的

opt_fin_cols = ['R_np_atoopc@xbx_ttm', 'B_total_equity_atoopc@xbx']  # 输出的财务字段，需要保留的


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
    # ROE：净资产收益率 = 净利润 / 净资产
    # R_np_atoopc@xbx_ttm:利润表的归属于母公司所有者的净利润ttm
    # B_total_equity_atoopc@xbx:资产负债表_所有者权益的归属于母公司所有者权益合计
    data['ROE'] = data['R_np_atoopc@xbx_ttm'] / data['B_total_equity_atoopc@xbx']
    exg_dict['ROE'] = 'last'

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



