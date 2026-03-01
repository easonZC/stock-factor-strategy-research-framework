"""
Helper functions for factor IC/group/style/industry/size analytics.
"""

import datetime
import numpy as np
import pandas as pd
import scipy
import gc
import math
import warnings

warnings.filterwarnings('ignore')


def float_num_process(num, return_type=float, keep=2, max=5):
    """
    针对绝对值小于1的数字进行特殊处理，保留非0的N位（N默认为2，即keep参数）
    输入  0.231  输出  0.23
    输入  0.0231  输出  0.023
    输入  0.00231  输出  0.0023
    如果前面max个都是0，直接返回0.0
    :param num: 输入的数据
    :param return_type: 返回的数据类型，默认是float
    :param keep: 需要保留的非零位数
    :param max: 最长保留多少位
    :return:
        返回一个float或str
    """

    # 如果输入的数据是0，直接返回0.0
    if num == 0.:
        return 0.0

    # 绝对值大于1的数直接保留对应的位数输出
    if abs(num) > 1:
        return round(num, keep)
    # 获取小数点后面有多少个0
    zero_count = -int(math.log10(abs(num)))
    # 实际需要保留的位数
    keep = min(zero_count + keep, max)

    # 如果指定return_type是float，则返回float类型的数据
    if return_type == float:
        return round(num, keep)
    # 如果指定return_type是str，则返回str类型的数据
    else:
        return str(round(num, keep))


def get_factor_by_period(folder, period_offset, target, need_shift, fa, keep_cols):
    '''
    读取数据的函数
    :param folder: 数据所在的文件夹路径
    :param period_offset: 根据period_offset
    :param target: 目标列名
    :param need_shift: 目标列是否需要shift
    :param fa: 因子计算目录名
    :param keep_cols: 读取数据后需要保存的列
    :return:
        返回读取到的所有数据
    '''

    print('正在读取并整理数据...')
    start_date = datetime.datetime.now()  # 记录开始时间

    base_path = folder + f'/周期数据/{period_offset}/基础数据/基础数据.pkl'
    df = pd.read_pickle(base_path)
    factors = [fa, '风格因子']
    common_cols = ['股票代码', '交易日期', '开盘价', '最高价', '最低价', '收盘价', '成交量', '成交额', '流通市值',
                   '总市值', '沪深300成分股', '上证50成分股',
                   '中证500成分股', '中证1000成分股', '中证2000成分股', '创业板指成分股', '新版申万一级行业名称',
                   '新版申万二级行业名称', '新版申万三级行业名称',
                   '09:35收盘价', '09:45收盘价', '09:55收盘价', '复权因子', '收盘价_复权', '开盘价_复权', '最高价_复权',
                   '最低价_复权']

    for fa in factors:

        factors_path = folder + f'/周期数据/{period_offset}/{fa}/{fa}.pkl'
        fa_df = pd.read_pickle(factors_path)

        # 对比一下前后列名是否有重复的
        repeat_cols = list(set(df.columns).intersection(set(fa_df.columns)))
        # 要排除掉交易日期和股票代码两列
        repeat_cols = [col for col in repeat_cols if col not in ['股票代码', '交易日期']]
        if len(repeat_cols) > 0:
            for col in repeat_cols:
                if col in common_cols:  # 如果是公共列，则删除
                    fa_df.drop(columns=[col], inplace=True)
                else:
                    print(f'{fa}文件中的{col}列与已经加载的数据重名，程序已经自动退出，请检查因子重名的情况后重新运行')
                    raise Exception(
                        f'{fa}文件中的{col}列与已经加载的数据重名，程序已经自动退出，请检查因子重名的情况后重新运行')
        df = pd.merge(df, fa_df, on=['交易日期', '股票代码'], how='left')

    # 删除必要字段为空的部分
    df = df.dropna(subset=keep_cols, how='any')
    # 筛选出风格因子列
    style_cols = [col for col in df.columns if '风格因子_' in col]
    # 取出部分列数据
    df = df[keep_cols + style_cols]
    gc.collect()  # 内存回收
    # target列是否需要shift
    if need_shift:
        df['下周期_' + target] = df.groupby('股票代码').apply(lambda x: x[target].shift()).reset_index(0)[target]
    # 加入offset列
    df['offset'] = period_offset.split('_')[-1]

    print(f'读取并整理数据完成，耗时：{datetime.datetime.now() - start_date}')

    return df


def offset_grouping(df, factor, bins=5):
    """
    对数据进行分层
    :param df:
    :param factor:
    :param bins:
    :return:
    """
    print("Starting offset_grouping...")
    print(f"Debug: factor name = '{factor}'")
    print(f"Debug: Available columns = {list(df.columns)}")
    print(f"Debug: Factor in columns = {factor in df.columns}")

    # 当周期股票数量
    df['当周期股票数'] = df.groupby(['交易日期'])['股票代码'].transform('count')
    df = df[df['当周期股票数'] >= bins].copy()

    print(f"Data shape before grouping: {df.shape}")
    if not df.empty:
        print(f"Sample factor values:\n{df[[factor]].head()}")

    factor_to_rank = factor
    # 检查选择因子是否会导致DataFrame（即重复列）
    if isinstance(df[factor], pd.DataFrame):
        print(f"Warning: Duplicate columns found for factor '{factor}'.")
        # 为排名操作创建一个新的唯一列名
        unique_factor_col = f"{factor}_for_ranking"
        # 取重复列中的第一列并将其分配给新的唯一名称
        df[unique_factor_col] = df[factor].iloc[:, 0]
        # 更新用于排名的因子名称
        factor_to_rank = unique_factor_col
        print(f"Debug: Using first column, stored as '{factor_to_rank}' for ranking.")

    # 确保factor列是数值类型
    if factor_to_rank in df.columns:
        try:
            df[factor_to_rank] = pd.to_numeric(df[factor_to_rank], errors='coerce')
            print(f"Debug: Successfully converted '{factor_to_rank}' to numeric.")
        except Exception as e:
            print(f"Warning: Could not convert factor '{factor_to_rank}' to numeric. Error: {e}")
            # 如果转换失败，则返回，因为排名将失败
            return df
    else:
        print(f"Error: Column '{factor_to_rank}' not found for numeric conversion.")
        # 如果列不存在，则返回
        return df

    # 根据factor计算因子的排名
    df['因子_排名'] = df.groupby(['交易日期'])[factor_to_rank].rank(ascending=True, method='first')
    # 根据因子的排名进行分组
    df['groups'] = df.groupby(['交易日期'])['因子_排名'].transform(
        lambda x: pd.qcut(x, q=bins, labels=range(1, bins + 1), duplicates='drop'))

    # 如果创建了临时列，则清理它
    if factor_to_rank != factor:
        df = df.drop(columns=[factor_to_rank])
        print(f"Debug: Removed temporary column '{factor_to_rank}'.")

    return df


def get_IC(df, factor, target, offset):
    '''
    计算IC等一系列指标
    :param df: 数据
    :param factor: 因子列名：测试的因子名称
    :param target: 目标列名：计算IC时的下周期数据
    :param offset: 周期
    :return:
    '''
    factor_col = df[factor]
    if isinstance(factor_col, pd.DataFrame):
        print(f"Warning: Duplicate columns found for factor '{factor}' in get_IC. Using first column.")
        factor_col = factor_col.iloc[:, 0]

    # 确保目标列也是Series
    target_col = df[target]
    if isinstance(target_col, pd.DataFrame):
        print(f"Warning: Duplicate columns found for target '{target}' in get_IC. Using first column.")
        target_col = target_col.iloc[:, 0]

    # 将因子和目标列合并，以便于分组
    temp_df = pd.concat([factor_col.rename('factor'), target_col.rename('target'), df['交易日期']], axis=1)

    # 按交易日期分组并计算相关性
    IC = temp_df.groupby('交易日期').apply(lambda x: x['factor'].corr(x['target'], method='spearman')).to_frame()
    IC.rename(columns={0: 'RankIC'}, inplace=True)
    IC.reset_index(inplace=True)
    IC['offset'] = offset
    return IC


def get_group_nv(df, next_ret, b_rate, s_rate, offset):
    """
    针对分组数据进行分析，给出分组的资金曲线、分箱图以及各分组的未来资金曲线
    :param df: 输入的数据
    :param next_ret: 未来涨跌幅的list
    :param b_rate: 买入手续费率
    :param s_rate: 卖出手续费率
    :param offset: 当前执行的是哪个offset的数据
    :return:
        返回分组资金曲线、分组持仓走势数据
    """

    print('正在进行因子分组分析...')
    start_date = datetime.datetime.now()  # 记录开始时间

    # 由于会对原始的数据进行修正，所以需要把数据copy一份
    temp = df.copy()

    # 将持仓周期的众数当做标准的持仓周期数
    temp['持仓周期'] = temp[next_ret].map(len)
    hold_nums = int(temp['持仓周期'].mode())
    temp[next_ret] = temp[next_ret].map(
        lambda x: x[: hold_nums] if len(x) > hold_nums else (x + [0] * (hold_nums - len(x))))

    # 计算下周期每天的净值，并扣除手续费得到下周期的实际净值
    temp['下周期每天净值'] = temp[next_ret].apply(lambda x: (np.array(x) + 1).cumprod())
    free_rate = (1 - b_rate) * (1 - s_rate)
    temp['下周期净值'] = temp['下周期每天净值'].apply(lambda x: x[-1] * free_rate)

    # 计算得到每组的资金曲线
    group_nv = temp.groupby(['交易日期', 'groups'])['下周期净值'].mean().reset_index()
    group_nv = group_nv.sort_values(by='交易日期').reset_index(drop=True)
    # 计算每个分组的累计净值
    group_nv['净值'] = group_nv.groupby('groups')['下周期净值'].cumprod()
    group_nv.drop('下周期净值', axis=1, inplace=True)

    # 计算当前数据有多少个分组
    bins = group_nv['groups'].max()

    # 计算各分组在持仓内的每天收益
    group_hold_value = pd.DataFrame(temp.groupby('groups')['下周期每天净值'].mean()).T
    # 所有分组的第一天都是从1开始的
    for col in group_hold_value.columns:
        group_hold_value[col] = group_hold_value[col].apply(lambda x: [1] + list(x))
    # 将未来收益从list展开成逐行的数据
    group_hold_value = group_hold_value.explode(list(group_hold_value.columns)).reset_index(drop=True).reset_index()
    # 重命名列
    group_cols = ['时间'] + [f'第{i}组' for i in range(1, bins + 1)]
    group_hold_value.columns = group_cols
    group_hold_value['offset'] = offset

    print(f'因子分组分析完成，耗时：{datetime.datetime.now() - start_date}')

    # 返回数据：分组资金曲线、分组持仓走势
    return group_nv, group_hold_value


def get_style_corr(df, factor, offset):
    '''
    计算风格暴露
    :param df:
    :param factor:
    :param offset:
    :return:
    '''
    factor_col = df[factor]
    if isinstance(factor_col, pd.DataFrame):
        print(f"Warning: Duplicate columns found for factor '{factor}' in get_style_corr. Using first column.")
        factor_col = factor_col.iloc[:, 0]

    style_factor_list = [col for col in df.columns if '风格因子' in col]
    if not style_factor_list:
        return pd.DataFrame()

    # 创建一个临时的DataFrame来进行分组和计算
    temp_df = pd.concat([factor_col.rename('factor'), df[style_factor_list], df['交易日期']], axis=1)
    
    # 分别计算每个风格因子的相关性，以避免广播错误
    correlations = {}
    for style_factor in style_factor_list:
        # 按交易日期分组，计算因子与单个风格因子的相关性，然后取均值
        corr = temp_df.groupby('交易日期').apply(lambda x: x['factor'].corr(x[style_factor], method='spearman')).mean()
        correlations[style_factor] = corr
        
    style_corr = pd.Series(correlations).to_frame()
    style_corr.rename(columns={0: '相关系数'}, inplace=True)
    style_corr.reset_index(inplace=True)
    style_corr.rename(columns={'index': '风格'}, inplace=True)
    style_corr['offset'] = offset
    return style_corr


def get_industry_data(df, factor, target, industry_col, industry_name_change):
    '''
    计算行业IC、行业暴露
    :param df:
    :param factor:
    :param target:
    :param industry_col:
    :param industry_name_change:
    :return:
    '''
    factor_col = df[factor]
    if isinstance(factor_col, pd.DataFrame):
        factor_col = factor_col.iloc[:, 0]
    target_col = df[target]
    if isinstance(target_col, pd.DataFrame):
        target_col = target_col.iloc[:, 0]

    # 计算每个行业在每个交易日期的IC
    temp_df = pd.concat([factor_col.rename('factor'), target_col.rename('target'), df[['交易日期', industry_col]]], axis=1)
    industry_IC = temp_df.groupby(['交易日期', industry_col]).apply(
        lambda x: x['factor'].corr(x['target'], method='spearman')
    ).reset_index(name='RankIC')

    # 计算整体平均IC（不区分交易日期），返回行业级IC
    industry_IC_avg = industry_IC.groupby(industry_col)['RankIC'].mean().reset_index()
    # 添加空的行业暴露列以便绘图
    industry_IC_avg['因子第一组选股在各行业的占比'] = 0
    industry_IC_avg['因子最后一组选股在各行业的占比'] = 0
    return industry_IC_avg


def get_market_value_data(df, factor, target, market_value, bins):
    '''
    计算市值IC、市值暴露
    '''
    factor_col = df[factor]
    if isinstance(factor_col, pd.DataFrame):
        factor_col = factor_col.iloc[:, 0]
    target_col = df[target]
    if isinstance(target_col, pd.DataFrame):
        target_col = target_col.iloc[:, 0]
    mv_col = df[market_value]
    if isinstance(mv_col, pd.DataFrame):
        mv_col = mv_col.iloc[:, 0]

    # 建立临时DataFrame
    temp_df = pd.concat([
        df['交易日期'],
        mv_col.rename('mv'),
        factor_col.rename('factor'),
        target_col.rename('target')], axis=1)
    # 分组计算每个市值分组下的相关性
    temp_df['mv_group'] = temp_df.groupby('交易日期')['mv'].transform(
        lambda x: pd.qcut(x, q=bins, labels=range(1, bins + 1), duplicates='drop'))
    mv_ic = temp_df.groupby(['交易日期', 'mv_group']).apply(
        lambda x: x['factor'].corr(x['target'], method='spearman')
    ).reset_index(name='RankIC')
    # 平均IC
    mv_ic_avg = mv_ic.groupby('mv_group')['RankIC'].mean().reset_index()
    mv_ic_avg.rename(columns={'mv_group': '市值分组'}, inplace=True)
    # 添加占比列
    mv_ic_avg['因子第一组选股在各市值分组的占比'] = 0
    mv_ic_avg['因子最后一组选股在各市值分组的占比'] = 0
    return mv_ic_avg


def IC_analysis(IC_list):
    '''
    整合各个offset的IC数据并计算相关的IC指标
    :param IC_list: 各个offset的IC数据
    :return:
        返回IC数据、IC字符串
    '''

    # 将各个offset的数据合并 并 整理
    IC = pd.concat(IC_list, axis=0)
    IC = IC.sort_values('交易日期').reset_index(drop=True)

    # 计算累计RankIC；注意：因为我们考虑了每个offset，所以这边为了使得各个不同period之间的IC累计值能够比较，故除以offset的数量
    IC['累计RankIC'] = IC['RankIC'].cumsum() / (len(IC['offset'].unique()))

    # ===计算IC的统计值，并进行约等
    # =IC均值
    IC_mean = float_num_process(IC['RankIC'].mean())
    # =IC标准差
    IC_std = float_num_process(IC['RankIC'].std())
    # =ICIR
    ICIR = float_num_process(IC_mean / IC_std)
    # =IC胜率
    # 如果累计IC为正，则计算IC为正的比例
    if IC['累计RankIC'].iloc[-1] > 0:
        IC_ratio = str(float_num_process((IC['RankIC'] > 0).sum() / len(IC) * 100)) + '%'
    # 如果累计IC为负，则计算IC为负的比例
    else:
        IC_ratio = str(float_num_process((IC['RankIC'] < 0).sum() / len(IC) * 100)) + '%'

    # 将上述指标合成一个字符串，加入到IC图中
    IC_info = f'IC均值：{IC_mean}，IC标准差：{IC_std}，ICIR：{ICIR}，IC胜率：{IC_ratio}'

    return IC, IC_info


def get_IC_month(IC):
    '''
    生成IC月历
    :param IC: IC数据
    :return:
        返回IC月历的df数据
    '''

    # resample到月份数据
    IC['交易日期'] = pd.to_datetime(IC['交易日期'])
    IC.set_index('交易日期', inplace=True)
    IC_month = IC.resample('M').agg({'RankIC': 'mean'})
    IC_month.reset_index(inplace=True)
    # 提取出年份和月份
    IC_month['年份'] = IC_month['交易日期'].dt.year.astype('str')
    IC_month['月份'] = IC_month['交易日期'].dt.month
    # 将年份月份设置为index，在将月份unstack为列
    IC_month = IC_month.set_index(['年份', '月份'])['RankIC']
    IC_month = IC_month.unstack('月份')
    IC_month.columns = IC_month.columns.astype(str)
    # 计算各月平均的IC
    IC_month.loc['各月平均', :] = IC_month.mean(axis=0)
    # 按年份大小排名
    IC_month = IC_month.sort_index(ascending=False)

    return IC_month


def group_analysis(group_nv_list, group_hold_value_list):
    '''
    针对分组数据进行分析，给出分组的资金曲线、分箱图以及各分组的未来资金曲线
    :param group_nv_list: 各个offset的分组净值数据
    :param group_hold_value_list: 各个offset的分组持仓走势数据
    :return:
        返回分组资金曲线、分箱图、分组持仓走势数据
    '''

    # 生成时间轴
    dates = []
    for group_nv in group_nv_list:
        dates.extend(list(set(group_nv['交易日期'].to_list())))
    time_df = pd.DataFrame(sorted(dates), columns=['交易日期'])

    # 遍历各个offset的资金曲线数据，合并到时间轴上，将合并后的数据append到列表中
    nv_list = []
    for group_nv in group_nv_list:
        group_nv = group_nv.groupby('groups').apply(
            lambda x: pd.merge(time_df, x, 'left', '交易日期').ffill())
        group_nv.reset_index(drop=True, inplace=True)
        nv_list.append(group_nv)

    # 将所有offset的分组资金曲线数据合并
    nv_df = pd.concat(nv_list, ignore_index=True)
    # 计算当前数据有多少个分组
    bins = nv_df['groups'].max()
    # 计算不同offset的每个分组的平均净值
    group_curve = nv_df.groupby(['交易日期', 'groups'])['净值'].mean().reset_index()
    # 将数据按照展开
    group_curve = group_curve.set_index(['交易日期', 'groups']).unstack().reset_index()
    # 重命名数据列
    group_cols = ['交易日期'] + [f'第{i}组' for i in range(1, bins + 1)]
    group_curve.columns = group_cols

    # 计算多空净值走势
    # 获取第一组的涨跌幅数据
    first_group_ret = group_curve['第1组'].pct_change()
    first_group_ret = first_group_ret.fillna(value=group_curve['第1组'].iloc[0] - 1)
    # 获取最后一组的涨跌幅数据
    last_group_ret = group_curve[f'第{bins}组'].pct_change()
    last_group_ret = last_group_ret.fillna(value=group_curve[f'第{bins}组'].iloc[0] - 1)
    # 判断到底是多第一组空最后一组，还是多最后一组空第一组
    if group_curve['第1组'].iloc[-1] > group_curve[f'第{bins}组'].iloc[-1]:
        ls_ret = (first_group_ret - last_group_ret) / 2
    else:
        ls_ret = (last_group_ret - first_group_ret) / 2
    # 计算多空净值曲线
    group_curve['多空净值'] = (ls_ret + 1).cumprod()
    # 计算绘制分箱所需要的数据
    group_value = group_curve[-1:].T[1:].reset_index()
    group_value.columns = ['分组', '净值']

    # 合并各个offset的持仓走势数据
    all_group_hold_value = pd.concat(group_hold_value_list, axis=0)
    # 取出需要求各个offset平均的列
    mean_cols = [col for col in all_group_hold_value.columns if '第' in col]
    # 新建空df
    group_hold_value = pd.DataFrame()
    # 设定时间列
    group_hold_value['时间'] = all_group_hold_value['时间'].unique()
    # 求各个组的mean
    for col in mean_cols:
        group_hold_value[col] = all_group_hold_value.groupby('时间')[col].mean()

    return group_curve, group_value, group_hold_value


def style_analysis(style_corr_list):
    '''
    计算因子的风格暴露
    :param style_corr_list: 各个offset的风格暴露数据
    :return:
       返回因子的风格暴露的数据
    '''

    # 合并各个offset的风格暴露数据
    style_corr = pd.concat(style_corr_list, axis=0)
    # 对各offset求平均
    style_corr = style_corr.groupby('风格')['相关系数'].mean().to_frame().reset_index()

    return style_corr


def industry_analysis(industry_data_list, industry_col):
    '''
    计算各个offset行业分析数据的平均值
    :param industry_data_list: 各个offset的行业分析数据
    :param industry_data_list: 行业列名
    :return:
        返回各个行业的RankIC数据、占比数据
    '''

    # 合并各个offset的数据 并 整理
    all_industry_data = pd.concat(industry_data_list, axis=0)
    all_industry_data = all_industry_data.groupby(industry_col).mean().reset_index()

    # 对每个行业求IC均值、行业占比第一组均值、行业占比最后一组均值
    industry_data = all_industry_data.groupby(industry_col).apply(
        lambda x: [x['RankIC'].mean(), x['因子第一组选股在各行业的占比'].mean(),
                   x['因子最后一组选股在各行业的占比'].mean()])
    industry_data = industry_data.to_frame().reset_index()  # 整理数据
    # 取出IC数据、行业占比_第一组数据、行业占比_最后一组数据
    industry_data['RankIC'] = industry_data[0].map(lambda x: x[0])
    industry_data['因子第一组选股在各行业的占比'] = industry_data[0].map(lambda x: x[1])
    industry_data['因子最后一组选股在各行业的占比'] = industry_data[0].map(lambda x: x[2])
    # 处理数据
    industry_data.drop(0, axis=1, inplace=True)
    # 以IC排序
    industry_data.sort_values('RankIC', ascending=False, inplace=True)

    return industry_data


def market_value_analysis(market_value_list):
    '''
    计算各个offset市值分析数据的平均值
    :param market_value_list: 各个offset的市值分析数据
    :return:
        返回各个市值分组的RankIC数据、占比数据
    '''

    # 合并各个offset的数据 并 整理
    all_market_value_data = pd.concat(market_value_list, axis=0)
    all_market_value_data = all_market_value_data.groupby('市值分组').mean().reset_index()

    # 对每个市值分组求IC均值、市值占比第一组均值、数字hi占比最后一组均值
    market_value_data = all_market_value_data.groupby('市值分组').apply(
        lambda x: [x['RankIC'].mean(), x['因子第一组选股在各市值分组的占比'].mean(),
                   x['因子最后一组选股在各市值分组的占比'].mean()])
    market_value_data = market_value_data.to_frame().reset_index()  # 整理数据
    # 取出IC数据、市值占比_第一组数据、市值占比_最后一组数据
    market_value_data['RankIC'] = market_value_data[0].map(lambda x: x[0])
    market_value_data['因子第一组选股在各市值分组的占比'] = market_value_data[0].map(lambda x: x[1])
    market_value_data['因子最后一组选股在各市值分组的占比'] = market_value_data[0].map(lambda x: x[2])
    # 处理数据
    market_value_data.drop(0, axis=1, inplace=True)
    # 以市值分组大小排序
    market_value_data.sort_index(ascending=True, inplace=True)

    return market_value_data


def auto_offset(period):
    res = [0]
    # 判断指定的period是否为int类型
    if isinstance(period, int):
        # 如果是int类型，则有int个offset
        res = list(range(0, period))
    # 判断指定的period是否为str类型
    elif isinstance(period, str):
        # 判断period中是否包含W
        if ('W' in period.upper()):
            # 如果period只有W，即 period == 'W'
            if len(period) == 1:
                res = [0, 1, 2, 3, 4]
            # 如果period为N个W，比如 period == '2W'，则有两个offset
            else:
                res = list(range(0, int(period[:-1])))
        # 判断period中是否包含M
        elif 'M' in period.upper():
            # 如果period == 'M'
            if len(period) == 1:
                res = [0]
            # 如果period 为 N个M，比如 period == '2M'，则有两个offset
            else:
                res = list(range(0, int(period[:-1])))

    return res



