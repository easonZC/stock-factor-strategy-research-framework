"""
Factor definition module: 净利润相关因子.
"""

name = file_name = __file__.replace('\\', '/').split('/')[-1].replace('.py', '')  # 当前文件的名字

ipt_fin_cols = ['R_np@xbx']  # 输入的财务字段，财务数据上原始的

opt_fin_cols = ['R_np@xbx_单季同比', 'R_np@xbx_单季环比', 'R_np@xbx_ttm同比']  # 输出的财务字段，需要保留的



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
    # R_np@xbx_ttm同比:利润表的净利润ttm同比
    # R_np@xbx_单季同比：利润表的净利润单季同比
    # R_np@xbx_单季环比：利润表的净利润单季环比

    # 净利润TTM_同比：当前净利润 TTM 与去年同期净利润 TTM 之间的对比。
    data['净利润TTM_同比'] = data['R_np@xbx_ttm同比']
    exg_dict['净利润TTM_同比'] = 'last'

    # 净利润TTM_同比增速：本季度净利润TTM_同比 - 上季度净利润TTM_同比
    data['净利润TTM_同比增速'] = data['R_np@xbx_ttm同比'] - data['R_np@xbx_ttm同比'].shift(60)
    exg_dict['净利润TTM_同比增速'] = 'last'

    # 净利润_单季_同比：当前季度的净利润与去年同季度净利润之间的百分比变化率
    data['净利润_单季_同比'] = data['R_np@xbx_单季同比']
    exg_dict['净利润_单季_同比'] = 'last'

    # 净利润_单季_环比：当前季度的净利润与上一季度净利润之间的百分比变化率
    data['净利润_单季_环比'] = data['R_np@xbx_单季环比']
    exg_dict['净利润_单季_环比'] = 'last'

    # 净利润_单季_同比增速：本季度净利润_单季_同比 - 上季度净利润_单季_同比
    data['净利润_单季_同比增速'] = data['R_np@xbx_单季同比'] - data['R_np@xbx_单季同比'].shift(60)
    exg_dict['净利润_单季_同比增速'] = 'last'

    # 净利润_单季_环比增速：本季度净利润_单季_环比 - 上季度净利润_单季_环比
    data['净利润_单季_环比增速'] = data['R_np@xbx_单季环比'] - data['R_np@xbx_单季环比'].shift(60)
    exg_dict['净利润_单季_环比增速'] = 'last'

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



