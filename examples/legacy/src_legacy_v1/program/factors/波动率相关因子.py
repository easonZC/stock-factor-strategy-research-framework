"""
Factor definition module: 波动率相关因子.
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
    # 上行波动率_N：最近N日收益率>0的部分的标准差
    # 下行波动率_N：最近N日收益率＜0的部分的标准差
    data['正收益率'] = data['涨跌幅'].apply(lambda x: x if x > 0 else 0)  # 筛选出收益率大于0的部分,小于等于0的部分用0填充
    data['负收益率'] = data['涨跌幅'].apply(lambda x: x if x <= 0 else 0)  # 筛选出收益率小于0的部分,大于等于0的部分用0填充
    for n in [5, 10, 20, 60, 120, 250]:
        data[f'上行波动率_{n}'] = data['正收益率'].rolling(n).std()
        exg_dict[f'上行波动率_{n}'] = 'last'

        data[f'下行波动率_{n}'] = data['负收益率'].rolling(n).std()
        exg_dict[f'下行波动率_{n}'] = 'last'

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



