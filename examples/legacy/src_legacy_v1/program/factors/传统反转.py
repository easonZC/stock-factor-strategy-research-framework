"""
Factor definition module: 传统反转.
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


def cal_factors(data, fin_data, fin_raw_data, exg_dict):
    """
    计算传统反转因子（1个月形成期）
    严格遵循广发证券文档定义：
    - 形成期：20个交易日（1个月）
    - 因子值：过去20日收益率取负值
    - 月度调仓（持有期约20日）
    
    :param data: DataFrame包含股票数据，必须有'收盘价'列
    :param exg_dict: 因子扩展字典
    :return: 包含反转因子的DataFrame
    """
    # 参数设置
    FORMATION_PERIOD =   5 # 形成期长度（季度调仓）

    # 计算形成期收益率
    # 方法：当前收盘价 / 60日前收盘价 - 1
    data['前收盘价'] = data.groupby('股票代码')['收盘价'].shift(FORMATION_PERIOD)
    data['形成期收益率'] = data['收盘价'] / data['前收盘价'] - 1
    
    # 计算反转因子值 = -形成期收益率
    data['反转_1M'] = -data['形成期收益率']
    
    # 设置扩展规则：使用最近一个交易日的因子值
    exg_dict['反转_1M'] = 'last'
    
    return data, exg_dict


