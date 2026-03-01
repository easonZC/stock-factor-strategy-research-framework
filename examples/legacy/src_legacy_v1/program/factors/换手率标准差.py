"""
Factor definition module: 换手率标准差.
"""

name = file_name = __file__.replace('\\', '/').split('/')[-1].replace('.py', '')

ipt_fin_cols = []  # 输入的财务字段
opt_fin_cols = []  # 输出的财务字段

import pandas as pd
import numpy as np


def special_data():
    return

def after_resample(data):
    return data

def cal_cross_factor(data):
    return data

def robust_calculate_turnover(data):

    data = data.copy()
    
    data['calc_成交额'] = data['成交量'] * data['收盘价'].replace(0, 1)  # 防除零
    cap_ratio = (data['calc_成交额'] / data['流通市值']).median()    
    if cap_ratio > 10:
        print(f"检测到流通市值单位问题(ratio={cap_ratio:.2f})，进行万元转元转换")
        data['流通市值_元'] = data['流通市值'] * 10000
    else:
        data['流通市值_元'] = data['流通市值']
    
    data['流通股本'] = data['流通市值_元'] / data['收盘价'].replace(0, np.nan)
    
    cap_threshold = np.percentile(data['流通股本'].dropna(), 95) 
    data.loc[data['流通股本'] > cap_threshold * 10, '流通股本'] = cap_threshold
    
    # 计算换手率
    data['换手率'] = data['成交量'] / data['流通股本'].replace(0, np.nan)
    data['换手率'] = data['换手率'].clip(lower=0, upper=1)  # 0-100%范围

    # 最终填充NaN
    data['换手率'] = data['换手率'].fillna(0.02)  # 中位换手率
    
    # 调试输出
    print(f"换手率统计：均值={data['换手率'].mean():.4f}，范围=[{data['换手率'].min():.4f}, {data['换手率'].max():.4f}]")
    
    return data

def calculate_std_turnover(data, period_months):
    """换手率标准差计算"""
    trading_days = int(period_months * 20)  # 每月20个交易日
    factor_name = f'std_turn_{period_months}m'
    
    print(f"计算{period_months}个月换手率波动性...")
    
    # 初始化因子列
    data[factor_name] = np.nan
    
    # 按股票分组计算
    for symbol, group in data.groupby('股票代码'):
        group = group.sort_values('交易日期')
        
        # 计算滚动标准差（至少5个数据点）
        for i in range(trading_days, len(group)):
            # 提取形成期窗口
            window_data = group.iloc[i-trading_days:i+1]
            
            # 计算标准差（至少5个非NaN值）
            valid_turnover = window_data['换手率'].dropna()
            if len(valid_turnover) >= 5:
                std_value = valid_turnover.std()
            else:
                std_value = 0.02  # 默认值
                
            data.loc[group.index[i], factor_name] = std_value
    
    # 数据不足时的处理
    data[factor_name] = data[factor_name].fillna(0.02)
    
    # 调试输出
    nan_count = data[factor_name].isna().sum()
    print(f"{factor_name}: NaN值数量={nan_count}, 有效比例={(len(data)-nan_count)/len(data)*100:.1f}%")
    
    return data



def cal_factors(data, fin_data, fin_raw_data, exg_dict):

    # Step 1: 计算换手率
    data = robust_calculate_turnover(data)
        
    # Step 2: 计算各周期波动性因子
    periods = [1, 3, 6]  # 1/3/6个月
        
    for months in periods:
        data = calculate_std_turnover(data, months)
        exg_dict[f'std_turn_{months}m'] = 'last' 
        
    
    return data, exg_dict


