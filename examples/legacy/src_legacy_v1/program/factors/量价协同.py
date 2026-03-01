"""
Factor definition module: 量价协同.
"""

name = file_name = __file__.replace('\\', '/').split('/')[-1].replace('.py', '')

ipt_fin_cols = []
opt_fin_cols = []

import pandas as pd
import numpy as np

def special_data():
    return
def after_resample(data):
    return data

def cal_cross_factor(data):
    return data

def cal_factors(data, fin_data, fin_raw_data, exg_dict):
    """
    修复版量价协同因子计算
    """
    print("===== 开始计算量价协同因子 =====")
    
    try:
        # 确保数据按股票代码和时间排序
        data = data.sort_values(['股票代码', '交易日期'])
        
        # 数据清洗：确保所有价格和成交量列都是数值类型
        numeric_columns = ['收盘价', '成交量', '最高价', '最低价', '开盘价']
        data = clean_numeric_data(data, numeric_columns)
        
        # 计算价格变化和成交量变化
        data['price_change'] = data.groupby('股票代码')['收盘价'].pct_change()
        data['volume_change'] = data.groupby('股票代码')['成交量'].pct_change()
        
        # 处理可能的NaN值
        data['price_change'] = data['price_change'].fillna(0)
        data['volume_change'] = data['volume_change'].fillna(0)
        
        # 定义安全的量价相关性计算函数
        def calculate_pv_correlation(group):
            # 确保使用数值数据
            price_changes = pd.to_numeric(group['price_change'], errors='coerce').dropna()
            volume_changes = pd.to_numeric(group['volume_change'], errors='coerce').dropna()
            
            # 确保长度一致
            min_len = min(len(price_changes), len(volume_changes))
            if min_len < 5:
                return 0.0
                
            price_changes = price_changes.iloc[-min_len:]
            volume_changes = volume_changes.iloc[-min_len:]
            
            # 计算相关性
            correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        
        # 使用更安全的方式计算滚动相关性
        data['pv_correlation'] = data.groupby('股票代码').apply(
            lambda x: x.rolling(window=10, min_periods=5).apply(
                lambda y: calculate_pv_correlation(pd.DataFrame({
                    'price_change': y['price_change'],
                    'volume_change': y['volume_change']
                })) if len(y) > 0 else 0,
                raw=False
            )
        ).reset_index(level=0, drop=True)
        
        # 计算量价背离
        data['pv_divergence'] = np.where(
            (data['price_change'] > 0) & (data['volume_change'] < 0),
            -1,
            np.where(
                (data['price_change'] < 0) & (data['volume_change'] > 0),
                1,
                0
            )
        )
        
        # 计算量价协同分数
        data['pv_synergy'] = data['pv_correlation'] + data['pv_divergence']
        
        # 注册因子
        exg_dict['pv_synergy'] = 'last'
        
        print("量价协同因子计算成功！")
        
    except Exception as e:
        print(f"量价协同因子计算失败: {e}")
        import traceback
        traceback.print_exc()
        # 确保列存在
        if 'pv_synergy' not in data.columns:
            data['pv_synergy'] = np.nan
    
    return data, exg_dict

def clean_numeric_data(data, numeric_columns):
    """
    清洗数值数据，确保所有指定列都是数值类型
    """
    for col in numeric_columns:
        if col in data.columns:
            # 转换为字符串后处理特殊值
            data[col] = data[col].astype(str)
            
            # 替换常见非数值字符
            data[col] = data[col].str.replace('N', '').str.replace('C', '')
            data[col] = data[col].str.replace('*', '').str.replace(' ', '')
            data[col] = data[col].str.replace('--', '0').str.replace('NaN', '0')
            
            # 转换为数值类型，无法转换的设为NaN
            data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # 填充NaN值
            data[col] = data[col].fillna(method='ffill').fillna(0)
    
    return data


