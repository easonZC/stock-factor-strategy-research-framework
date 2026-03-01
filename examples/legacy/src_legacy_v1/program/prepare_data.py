"""
Data-preparation pipeline that builds base data and factor datasets.
"""

import os.path
import time
import sys
import shutil
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from joblib import Parallel, delayed

from program import settings as Cfg
from program import runtime_utils as Fun

# 可以从程序外部指定到底跑【live_strategies】还是【research_strategies】下的策略
# 也就是说即便Config里面设置的跑【research_strategies】，外部程序也将其改为【live_strategies】
if len(sys.argv) > 1:
    Cfg.stg_folder = sys.argv[1]

# ===脚本运行需要设置的变量
# 是否需要多线程处理，True表示多线程，False表示单线程
multiple_process = True


def merge_factor_pickles(factor_dir, factor_name, output_filename):
    """
    创建聚合的pickle文件,将个股文件合并成一个文件
    
    Args:
        factor_dir: 包含个股.pkl文件的目录
        factor_name: 因子名称（用于筛选文件）
        output_filename: 输出聚合文件的名称
    
    Returns:
        bool: 是否成功创建聚合文件
    """
    print(f"创建聚合文件: {factor_dir}/{output_filename}")
    
    if not os.path.exists(factor_dir):
        print(f"警告: 目录不存在 {factor_dir}")
        return False
    
    # 获取该因子的个股文件列表
    file_list = Fun.get_file_in_folder(factor_dir, '.pkl', filters=[factor_name])
    
    if len(file_list) == 0:
        print(f"警告: 在 {factor_dir} 中没有找到 {factor_name} 相关的文件")
        return False
    
    print(f"找到 {len(file_list)} 个 {factor_name} 文件，开始合并...")
    
    # 并行读取所有个股文件
    df_list = Parallel(Cfg.n_job)(
        delayed(pd.read_pickle)(os.path.join(factor_dir, code)) for code in file_list
    )
    
    # 合并所有DataFrame
    all_df = pd.concat(df_list, ignore_index=True)
    
    # 保存聚合文件
    save_path = os.path.join(factor_dir, output_filename)
    all_df.reset_index(drop=True).to_pickle(save_path)
    print(f"成功保存聚合文件: {save_path}，数据形状: {all_df.shape}")
    
    return True

def create_aggregated_file(factor_dir, factor_name, output_filename):
    """Backward-compatible alias for legacy scripts."""
    return merge_factor_pickles(factor_dir, factor_name, output_filename)

# ===循环读取并且合并
def process_single_symbol(code):
    """
    整理数据核心函数
    :param code: 股票代码
    :return: 一个包含该股票所有历史数据的DataFrame
    """
    try:
        print(code, '开始计算')
        # =读入股票数据
        path = os.path.join(Cfg.stock_data_path, code)
        df = pd.read_csv(path, encoding='gbk', skiprows=1, parse_dates=['交易日期'])


        # 【重要修改】筛选逻辑已前置，此处不再需要单独筛选
        
        # =计算涨跌幅
        df['涨跌幅'] = df['收盘价'] / df['前收盘价'] - 1
        # 为之后开盘买入做好准备
        df['开盘买入涨跌幅'] = df['收盘价'] / df['开盘价'] - 1
        # 计算换手率
        df['换手率'] = df['成交额'] / df['流通市值']

        # =计算复权价：计算所有因子当中用到的价格，都使用复权价
        df = Fun.cal_fuquan_price(df, fuquan_type='后复权')

        # =计算交易天数
        df['上市至今交易天数'] = df.index.astype('int') + 1

        # 获取股票基础数据的路径
        stock_base_path = os.path.join(Cfg.factor_path, '日频数据/基础数据/', code.replace('csv', 'pkl'))

        # 对数据进行预处理：合并指数、计算下个交易日状态、基础数据resample等
        total_df, new_total = Fun.pre_process(stock_base_path, subset_day, cut_day, df, index_data, po_list, per_oft_df,
                                              total_mode, Cfg.end_exchange)
        # 如果全量数据是空的，则直接返回
        if total_df.empty:
            return

        # =导入财务数据，将个股数据和财务数据合并，并计算需要的财务指标的衍生指标(所有的财务数据都重新合并一下)
        #total_df, fin_df, fin_raw_df = Fin.merge_with_finance_data(total_df, code[:-4], Cfg.fin_path, ipt_fin_cols,
        #                                                           opt_fin_cols, {})
        # Create empty DataFrames for financial data since we don't need it for basic strategies
        fin_df = pd.DataFrame()
        fin_raw_df = pd.DataFrame()
        
        # 是否是全量更新数据
        if total_mode:
            increment_df = pd.DataFrame(columns=total_df.columns)
        else:
            # 拿一下增量数据
            increment_df = pd.DataFrame(total_df[total_df['交易日期'] >= cut_day])


        # =遍历调用每个策略的load函数，合并其他数据
        for name, func_info in load_functions.items():
            # 遍历看看有没有那个因子是需要用全部数据才能算的
            # 是否用全量，因为可能有多个load函数，但凡其中一个需要用全量计算，全体都应该用全量来计算，所以初始值是0
            _is_total = 0
            for _fa in func_info['factors']:
                # 因子的保存路径
                _save_path = os.path.join(Cfg.factor_path, f'日频数据/{_fa}/', code.replace('csv', 'pkl'))
                # 通过相加来判断整体是否需要用全量计算
                _is_total += Fun.use_total(_save_path, is_folder=False)
            # 如果是全量计算
            if _is_total or total_mode:
                total_df = func_info['func'](total_df)
                new_total = True
            # 否则只用计算增量数据
            else:
                increment_df = func_info['func'](increment_df)

        # =遍历计算每一个因子
        for _fa in fa_info.keys():
            exg_dict = {}  # 重要变量，在将日线数据转换成周期数据时使用。key为需要转换的数据，对应的value为转换的方法，包括first,max,min,last,sum等.
            # 获取因子的保存路径
            _save_path = os.path.join(Cfg.factor_path, f'日频数据/{_fa}/')
            # 创建文件夹（如果无）
            os.makedirs(_save_path, exist_ok=True)
            # 日频因子的保存路径
            _save_path = os.path.join(_save_path, code.replace('csv', 'pkl'))
            # 是否需要使用全量计算
            _is_total = Fun.use_total(_save_path, is_folder=False)
            # 是否全量计算数据
            if _is_total or new_total or total_mode:
                # 计算因子
                total_df, exg_dict = fa_info[_fa]['cls'].cal_factors(total_df, fin_df, fin_raw_df, exg_dict)
                # 需要保存的因子列表
                factor_cols = list(exg_dict.keys())
                # 只保留需要的字段
                total_factor_df = total_df[['交易日期', '股票代码'] + factor_cols]
                # 全量数据直接保存
                total_factor_df.to_pickle(_save_path)
            else:
                # 计算因子
                increment_df, exg_dict = fa_info[_fa]['cls'].cal_factors(increment_df, fin_df, fin_raw_df, exg_dict)
                factor_cols = list(exg_dict.keys())  # 需要保存的因子列表
                # 只保留需要的字段
                increment_factor_df = increment_df[['交易日期', '股票代码'] + factor_cols]
                # 增量数据先读取再保存
                total_factor_df = pd.read_pickle(_save_path)
                _waring_info = f'股票：{code}，因子：{_fa}，日频数据前后结果对比不一致，请检查！！！'
                # 如果历史数据和增量数据的columns存在不一致，输出上述的报警信息，数据很可能有误
                total_factor_df = \
                    Fun.merge_with_hist(increment_factor_df, cut_day, total_factor_df, False, _waring_info)[1]
                # 全量数据直接保存
                total_factor_df.to_pickle(_save_path)

            # =遍历所有的周期，保存周期因子
            for _po in fa_info[_fa]['per_oft']:
                # 周期因子的保存路径
                _save_path = os.path.join(Cfg.factor_path, f'周期数据/{_po}/{_fa}')
                # 创建保存周期因子数据的文件夹
                os.makedirs(_save_path, exist_ok=True)
                _save_path = os.path.join(_save_path, code.replace('csv', 'pkl'))
                _is_total = Fun.use_total(_save_path, is_folder=False)
                # =是否全量转换数据
                if _is_total or new_total or total_mode:
                    # 全量数据转换成周期数据
                    period_df = Fun.transfer_factor_data(total_factor_df, per_oft_df, _po, exg_dict)
                    # 计算周期级别的因子
                    period_df = fa_info[_fa]['cls'].after_resample(period_df)
                    # 重置索引并保存
                    period_df.reset_index(drop=True).to_pickle(_save_path)
                else:
                    # 只对增量数据进行周期转换
                    period_df = Fun.transfer_factor_data(increment_factor_df, per_oft_df, _po, exg_dict)
                    # 计算周期级别的因子
                    period_df = fa_info[_fa]['cls'].after_resample(period_df)
                    _waring_info = f'股票：{code}，因子：{_fa}，周期数据前后结果对比不一致，请检查！！！'
                    # 如果历史数据和增量数据的columns存在不一致，输出上述的报警信息，数据很可能有误
                    period_df = Fun.merge_with_hist(period_df, cut_day, _save_path, True, _waring_info)[1]
                    # 重置索引并保存
                    period_df.reset_index(drop=True).to_pickle(_save_path)
    except Exception as err:
        print(code, '出现异常！！！！！！！！！！！！！！！！')
        raise err
    return

def calculate_by_stock(code):
    """Backward-compatible alias for legacy scripts."""
    return process_single_symbol(code)

if __name__ == '__main__':
    print("=== 数据整理脚本开始 ===")
    
    # === 清理之前的因子数据 ===
    print("清理之前的因子数据...")
    factor_data_path = os.path.join(Cfg.root_path, 'data/因子数据')
    if os.path.exists(factor_data_path):
        try:
            shutil.rmtree(factor_data_path)
            print(f"已删除因子数据目录: {factor_data_path}")
        except Exception as e:
            print(f"删除因子数据目录时出错: {e}")
    
    # 重新创建因子数据目录
    os.makedirs(factor_data_path, exist_ok=True)
    print("重新创建因子数据目录")
    
    now = time.time()  # 用于记录运行时间

    # 如果要尾盘换仓，需要先从行情网站拿一下数据
    if Cfg.end_exchange:
        python_exe = sys.executable
        os.system('%s %s/program/0_尾盘获取数据.py' % (python_exe, Cfg.root_path))

    print("开始因子检查...")

    # ===读取：股票列表、指数数据、策略信息、周期表等
    # 1.读取所有股票代码的列表
    all_stock_files = Fun.get_file_in_folder(
        Cfg.stock_data_path,  # 股票数据路径
        '.csv'                # 文件后缀
    )
    stock_code_list =[code for code in all_stock_files ]  # 默认使用所有股票代码
    '''
    # 2.只筛选出以'bj'开头的北交所股票
    stock_code_list = [code for code in all_stock_files if code.startswith('bj')]
    
    print(f'发现 {len(all_stock_files)} 个股票文件，筛选出 {len(stock_code_list)} 个北交所股票进行处理。')

    if not stock_code_list:
        print("错误：未找到任何北交所股票（代码以'bj'开头），程序将退出。请检查股票数据路径或文件名。")
        sys.exit() # 如果列表为空，直接退出
    '''
    # 3.导入上证指数，保证指数数据和股票数据在同一天结束，不然会出现问题。
    index_data = Fun.import_index_data(Cfg.index_path, back_trader_end=Cfg.date_end)
    print(f'从指数获取最新交易日：{index_data["交易日期"].iloc[-1].strftime("%Y-%m-%d")}')
    # 4.导入策略文件夹中的所有选股策略
    stg_file = Fun.get_file_in_folder(os.path.join(Cfg.root_path, f'program/{Cfg.stg_folder}/'), '.py',
                                      filters=['_init_'], drop_type=True)
    # 5.加载所有周期所有offset的df
    per_oft_df = Fun.read_period_and_offset_file(Cfg.period_offset_file)

    # ===获取运行信息
    # 本函数输入：策略文件、周期表、是否回测模式、交易日期，其他因子
    # 输出：因子运行信息fa_info、原始财务字段ipt_fin_cols、输出的财务字段，外部数据函数合集load_functions，周期列表po_list
    # 详细内容建议点进函数内部查看
    fa_info, ipt_fin_cols, opt_fin_cols, load_functions, po_list = Fun.get_run_info(stg_file, per_oft_df,
                                                                                    Cfg.run_all_offset,
                                                                                    index_data['交易日期'].iloc[-1],
                                                                                    Cfg.other_factors,
                                                                                    Cfg.stg_folder)
    # 打印当前的回测模式
    print(f'{"回测模式 所有offset都将运行" if Cfg.run_all_offset else "实盘模式,仅对应offset将被运行"}')

    # 运行每个因子的special_data函数
    for factor in fa_info.keys():
        fa_info[factor]['cls'].special_data()

    # 判断当前是增量模式还是存量模式，Cfg.subset_num是具体的数字时是增量模式，为None时是全量数据
    if Cfg.subset_num:  # 增量模式
        # 增量模式下，subset_num的数值要大于你最大参数的1.9倍会比较稳妥
        # subset_day用于切片算因子
        subset_day = index_data['交易日期'].iloc[-int(Cfg.subset_num)] # 强制转换为整数
        # cut_day用于合并数据
        cut_day_offset = math.ceil(Cfg.subset_num * 0.3) # 使用ceil确保向上取整
        cut_day = index_data['交易日期'].iloc[-cut_day_offset]
        # 打印增量计算的信息
        print(f'增量计算信息：个股数据计算从{subset_day.strftime("%Y-%m-%d")}开始，'f'合并数据从{cut_day.strftime("%Y-%m-%d")}开始')
        # 关闭强制增量模式
        total_mode = False
    else:
        # 全量计算
        print('=====当前为全量计算模式=====')
        # 全量计算下，subset_day和cut_day都定义为全量数据最早的时间。
        subset_day = index_data['交易日期'].iloc[0]
        cut_day = index_data['交易日期'].iloc[0]
        # 开启强制增量模式
        total_mode = True

    # ===遍历每个股票，计算各个股票的因子（并行或者串行）
    # 并行计算，速度快但cpu负载高，速度和负载取决于config.n_job的大小
    if multiple_process:
        Parallel(Cfg.n_job)(delayed(process_single_symbol)(code) for code in stock_code_list)
    # 串行计算，一个一个算，速度慢但负载低
    else:
        for stock in stock_code_list:
            process_single_symbol(stock)

    # 遍历每个因子，计算其截面数据并创建聚合文件
    for fa in fa_info.keys():
        # 遍历该因子所有的周期
        for po in fa_info[fa]['per_oft']:
            # 周期因子数据路径
            period_factor_path = os.path.join(Cfg.factor_path, f'周期数据/{po}/{fa}/')
            # 创建聚合文件（先从个股文件合并）
            merge_factor_pickles(period_factor_path, fa, f'{fa}.pkl')
            
            # 读取刚创建的聚合文件
            save_path = os.path.join(period_factor_path, f'{fa}.pkl')
            if os.path.exists(save_path):
                all_df = pd.read_pickle(save_path)
                
                # 判断是否需要保存全部文件
                is_total = Fun.use_total(save_path, False)
                # 是否需要保存全部的数据
                if is_total or total_mode:
                    # 计算截面因子
                    all_df = fa_info[fa]['cls'].cal_cross_factor(all_df)
                    # 保存数据
                    all_df.reset_index(drop=True).to_pickle(save_path)
                else:
                    # 需要计算的增量数据
                    increment_all_df = all_df[all_df['交易日期'] >= subset_day]
                    # 计算截面因子
                    increment_all_df = fa_info[fa]['cls'].cal_cross_factor(increment_all_df)
                    # 合并异常的警告信息
                    waring_info = f'因子：{fa}，截面数据前后结果对比不一致，请检查！！！'
                    # 如果历史数据和增量数据的columns存在不一致，输出上述的报警信息，意味着数据错误建议重跑全量模式
                    all_df = Fun.merge_with_hist(increment_all_df, cut_day, save_path, True, waring_info)[1]
                    # 保存数据
                    all_df.reset_index(drop=True).to_pickle(save_path)

    # 基本数据也需要保存一份截面数据
    for po in po_list:
        # 周期基本数据的路径
        period_factor_path = os.path.join(Cfg.factor_path, f'周期数据/{po}/基础数据/')
        # 创建基础数据的聚合文件
        merge_factor_pickles(period_factor_path, '基础数据', '基础数据.pkl')
    # 打印一下数据整理的整体耗时
    print('耗时：', time.time() - now)






