"""
Factor definition module: 风格因子.
"""

import numpy as np

name = file_name = __file__.replace('\\', '/').split('/')[-1].replace('.py', '')  # 当前文件的名字

ipt_fin_cols = ['R_np@xbx', 'R_revenue@xbx', 'R_op@xbx', 'B_total_equity_atoopc@xbx', 'B_total_liab@xbx',
                'B_actual_received_capital@xbx', 'B_preferred_shares@xbx', 'B_total_assets@xbx',
                'B_total_equity_atoopc@xbx', 'B_total_liab_and_owner_equity@xbx']  # 输入的财务字段，财务数据上原始的

opt_fin_cols = ['R_np@xbx_ttm', 'B_total_equity_atoopc@xbx', 'R_revenue@xbx_ttm', 'R_np@xbx_ttm同比',
                'R_revenue@xbx_ttm同比', 'R_np@xbx_单季同比', 'R_revenue@xbx_单季同比', 'B_total_liab@xbx',
                'B_actual_received_capital@xbx', 'B_preferred_shares@xbx', 'B_total_assets@xbx',
                'B_total_equity_atoopc@xbx', 'B_total_liab_and_owner_equity@xbx', 'R_op@xbx_ttm']  # 输出的财务字段，需要保留的


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
    exg_dict['新版申万一级行业名称'] = 'last'
    exg_dict['总市值'] = 'last'

    # ===估值因子 (需要财务数据，暂时注释掉)
    # data[name + 'EP'] = data['R_np@xbx_ttm'] / data['总市值']  # 市盈率倒数
    # data[name + 'BP'] = data['B_total_equity_atoopc@xbx'] / data['总市值']  # 市净率倒数
    # data[name + 'SP'] = data['R_revenue@xbx_ttm'] / data['总市值']  # 市销率倒数
    # exg_dict[name + 'EP'] = 'last'
    # exg_dict[name + 'BP'] = 'last'
    # exg_dict[name + 'SP'] = 'last'

    # ===动量因子 (不需要财务数据)
    data[name + 'Ret_252'] = data['收盘价'].shift(21) / data['收盘价'].shift(252) - 1
    exg_dict[name + 'Ret_252'] = 'last'

    # ===反转因子 (不需要财务数据)
    data[name + 'Ret_21'] = data['收盘价'] / data['收盘价'].shift(21) - 1
    exg_dict[name + 'Ret_21'] = 'last'

    # ===成长因子 (需要财务数据，暂时注释掉)
    # data[name + '净利润ttm同比'] = data['R_np@xbx_ttm同比']
    # data[name + '营业收入ttm同比'] = data['R_revenue@xbx_ttm同比']
    # data[name + '净利润单季同比'] = data['R_np@xbx_单季同比']
    # data[name + '营业收入单季同比'] = data['R_revenue@xbx_单季同比']
    # exg_dict[name + '净利润ttm同比'] = 'last'
    # exg_dict[name + '营业收入ttm同比'] = 'last'
    # exg_dict[name + '净利润单季同比'] = 'last'
    # exg_dict[name + '营业收入单季同比'] = 'last'

    # ===杠杆因子 (需要财务数据，暂时注释掉)
    # data[name + 'MLEV'] = (data['总市值'] + data['B_total_liab@xbx']) / data['总市值']
    # data[name + 'BLEV'] = (data[['B_actual_received_capital@xbx', 'B_preferred_shares@xbx']].sum(axis=1, skipna=True)) / \
    #                       data['总市值']
    # data[name + 'DTOA'] = data['B_total_liab@xbx'] / data['B_total_assets@xbx']
    # exg_dict[name + 'MLEV'] = 'last'
    # exg_dict[name + 'BLEV'] = 'last'
    # exg_dict[name + 'DTOA'] = 'last'

    # ===波动因子 (不需要财务数据)
    data[name + 'Std21'] = data['涨跌幅'].rolling(21).std()
    data[name + 'Std252'] = data['涨跌幅'].rolling(252).std()
    exg_dict[name + 'Std21'] = 'last'
    exg_dict[name + 'Std252'] = 'last'

    # ===流动性因子 (不需要财务数据)
    data[name + '换手率5'] = data['换手率'].rolling(5).mean()
    data[name + '换手率10'] = data['换手率'].rolling(10).mean()
    data[name + '换手率20'] = data['换手率'].rolling(20).mean()
    exg_dict[name + '换手率5'] = 'last'
    exg_dict[name + '换手率10'] = 'last'
    exg_dict[name + '换手率20'] = 'last'

    # ===盈利因子 (需要财务数据，暂时注释掉)
    # data[name + 'ROE'] = data['R_np@xbx_ttm'] / data['B_total_equity_atoopc@xbx']  # ROE 净资产收益率
    # data[name + 'ROA'] = data['R_np@xbx_ttm'] / data['B_total_liab_and_owner_equity@xbx']  # ROA 资产收益率
    # data[name + '净利润率'] = data['R_np@xbx_ttm'] / data['R_revenue@xbx_ttm']  # 净利润率：净利润 / 营业收入
    # data[name + 'GP'] = data['R_op@xbx_ttm'] / data['B_total_assets@xbx']
    # exg_dict[name + 'ROE'] = 'last'
    # exg_dict[name + 'ROA'] = 'last'
    # exg_dict[name + '净利润率'] = 'last'
    # exg_dict[name + 'GP'] = 'last'

    # ===规模因子 (不需要财务数据)
    data[name + '总市值'] = np.log(data['总市值'])
    exg_dict[name + '总市值'] = 'last'

    # Note: The final style factors (风格因子_动量, 风格因子_反转, etc.) 
    # will be created in cal_cross_factor function and don't need to be in exg_dict here

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
    # ===估值 (需要财务数据，暂时注释掉)
    # data[name + 'EP排名'] = data.groupby('交易日期')[name + 'EP'].rank(ascending=True, method='min')
    # data[name + 'BP排名'] = data.groupby('交易日期')[name + 'BP'].rank(ascending=True, method='min')
    # data[name + 'SP排名'] = data.groupby('交易日期')[name + 'SP'].rank(ascending=True, method='min')
    # data[name + '_估值'] = data[name + 'EP排名'] + data[name + 'BP排名'] + data[name + 'SP排名']
    # data.drop(columns=[name + 'EP排名', name + 'BP排名', name + 'SP排名', name + 'EP', name + 'BP', name + 'SP'],
    #           inplace=True)

    # ===动量 (不需要财务数据)
    data[name + 'Ret_252排名'] = data.groupby('交易日期')[name + 'Ret_252'].rank(ascending=True, method='min')
    data[name + '_动量'] = data[name + 'Ret_252排名']
    data.drop(columns=[name + 'Ret_252排名', name + 'Ret_252'], inplace=True)

    # ===反转 (不需要财务数据)
    data[name + 'Ret_21排名'] = data.groupby('交易日期')[name + 'Ret_21'].rank(ascending=True, method='min')
    data[name + '_反转'] = data[name + 'Ret_21排名']
    data.drop(columns=[name + 'Ret_21排名', name + 'Ret_21'], inplace=True)

    # ===成长 (需要财务数据，暂时注释掉)
    # data[name + '净利润ttm同比排名'] = data.groupby('交易日期')[name + '净利润ttm同比'].rank(ascending=True, method='min')
    # data[name + '营业收入ttm同比排名'] = data.groupby('交易日期')[name + '营业收入ttm同比'].rank(ascending=True,
    #                                                                            method='min')
    # data[name + '净利润单季同比排名'] = data.groupby('交易日期')[name + '净利润单季同比'].rank(ascending=True, method='min')
    # data[name + '营业收入单季同比排名'] = data.groupby('交易日期')[name + '营业收入单季同比'].rank(ascending=True,
    #                                                                          method='min')
    # data[name + '_成长'] = data[name + '净利润ttm同比排名'] + data[name + '营业收入ttm同比排名'] + data[
    #     name + '净利润单季同比排名'] + data[name + '营业收入单季同比排名']
    # data.drop(columns=[name + '净利润ttm同比排名', name + '营业收入ttm同比排名', name + '净利润单季同比排名',
    #                    name + '营业收入单季同比排名', name + '净利润ttm同比', name + '营业收入ttm同比', name + '净利润单季同比',
    #                    name + '营业收入单季同比'], inplace=True)

    # ===杠杆 (需要财务数据，暂时注释掉)
    # data[name + 'MLEV排名'] = data.groupby('交易日期')[name + 'MLEV'].rank(ascending=True, method='min')
    # data[name + 'BLEV排名'] = data.groupby('交易日期')[name + 'BLEV'].rank(ascending=True, method='min')
    # data[name + 'DTOA排名'] = data.groupby('交易日期')[name + 'DTOA'].rank(ascending=True, method='min')
    # data[name + '_杠杆'] = data[name + 'MLEV排名'] + data[name + 'BLEV排名'] + data[name + 'DTOA排名']
    # data.drop(
    #     columns=[name + 'MLEV排名', name + 'BLEV排名', name + 'DTOA排名', name + 'MLEV', name + 'BLEV', name + 'DTOA'],
    #     inplace=True)

    # ===波动 (不需要财务数据)
    data[name + 'Std21排名'] = data.groupby('交易日期')[name + 'Std21'].rank(ascending=True, method='min')
    data[name + 'Std252排名'] = data.groupby('交易日期')[name + 'Std252'].rank(ascending=True, method='min')
    data[name + '_波动'] = data[name + 'Std21排名'] + data[name + 'Std252排名']
    data.drop(columns=[name + 'Std21排名', name + 'Std252排名', name + 'Std21', name + 'Std252'], inplace=True)

    # ===流动性 (不需要财务数据)
    data[name + '换手率5排名'] = data.groupby('交易日期')[name + '换手率5'].rank(ascending=True, method='min')
    data[name + '换手率10排名'] = data.groupby('交易日期')[name + '换手率10'].rank(ascending=True, method='min')
    data[name + '换手率20排名'] = data.groupby('交易日期')[name + '换手率20'].rank(ascending=True, method='min')
    data[name + '_流动性'] = data[name + '换手率5排名'] + data[name + '换手率10排名'] + data[name + '换手率20排名']
    data.drop(columns=[name + '换手率5排名', name + '换手率10排名', name + '换手率20排名', name + '换手率5',
                       name + '换手率10', name + '换手率20'], inplace=True)

    # ===盈利 (需要财务数据，暂时注释掉)
    # data[name + 'ROE排名'] = data.groupby('交易日期')[name + 'ROE'].rank(ascending=True, method='min')
    # data[name + 'ROA排名'] = data.groupby('交易日期')[name + 'ROA'].rank(ascending=True, method='min')
    # data[name + '净利润率排名'] = data.groupby('交易日期')[name + '净利润率'].rank(ascending=True, method='min')
    # data[name + 'GP排名'] = data.groupby('交易日期')[name + 'GP'].rank(ascending=True, method='min')
    # data[name + '_盈利'] = data[name + 'ROE排名'] + data[name + 'ROA排名'] + data[name + '净利润率排名'] + data[name + 'GP排名']
    # data.drop(columns=[name + 'ROE排名', name + 'ROA排名', name + '净利润率排名', name + 'GP排名',
    #                    name + 'ROE', name + 'ROA', name + '净利润率', name + 'GP'], inplace=True)

    # ===规模 (不需要财务数据)
    data[name + '总市值排名'] = data.groupby('交易日期')[name + '总市值'].rank(ascending=True, method='min')
    data[name + '_规模'] = data[name + '总市值排名']
    data.drop(columns=[name + '总市值排名', name + '总市值'], inplace=True)
    return data



