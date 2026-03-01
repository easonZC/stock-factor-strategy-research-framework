"""
Factor definition module: RSI.
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
    Calculate the Relative Strength Index (RSI) for the given data.
    :param data: DataFrame containing stock data with a '收盘价' column.
    :param window: The rolling window size for RSI calculation (default is 14).
    :return: DataFrame with an added 'RSI' column.
    """
    window = 14
    # Calculate daily price changes
    data['价格变化'] = data['收盘价'].diff()

    # Separate gains and losses
    data['涨幅'] = data['价格变化'].apply(lambda x: x if x > 0 else 0)
    data['跌幅'] = data['价格变化'].apply(lambda x: -x if x < 0 else 0)

    # Calculate average gains and losses
    data['平均涨幅'] = data['涨幅'].rolling(window=window, min_periods=1).mean()
    data['平均跌幅'] = data['跌幅'].rolling(window=window, min_periods=1).mean()

    # Compute Relative Strength (RS)
    RS = data['平均涨幅'] / data['平均跌幅']

    # Calculate RSI
    data['RSI'] = 100 - (100 / (1 + RS))

    exg_dict['RSI'] = 'last'

    return data, exg_dict


