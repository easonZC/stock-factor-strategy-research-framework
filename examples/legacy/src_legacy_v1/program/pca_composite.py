"""
PCA-based composite-factor builder for combining existing factor series.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
from joblib import Parallel, delayed

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from program import settings as Cfg

warnings.filterwarnings('ignore')

# =====配置参数=====
# 要合成的因子列表
source_factors = ['RSI', 'RVI']  # 源因子名称
target_factor = 'RSI_RVI_PCA'   # 目标复合因子名称
periods = ['W_0']               # 要处理的周期列表
n_components = 1                # PCA主成分数量，默认取第一主成分
multiple_process = True         # 是否使用多进程


def build_period_pca_factor(period):
    """
    为特定周期创建PCA复合因子
    
    Args:
        period: 周期名称，如 'W_0'
    """
    print(f"开始处理周期: {period}")
    
    # 构建因子数据路径
    period_path = os.path.join(Cfg.factor_path, f'周期数据/{period}')
    
    # 存储所有因子数据的字典
    factor_data_dict = {}
    
    # 读取各个源因子数据
    for factor_name in source_factors:
        factor_file_path = os.path.join(period_path, factor_name, f'{factor_name}.pkl')
        
        if not os.path.exists(factor_file_path):
            print(f"警告: 因子文件不存在 {factor_file_path}")
            print(f"请先运行 prepare_data.py 生成 {factor_name} 因子数据")
            return False
            
        print(f"读取因子数据: {factor_file_path}")
        factor_data = pd.read_pickle(factor_file_path)
        factor_data_dict[factor_name] = factor_data
        print(f"因子 {factor_name} 数据形状: {factor_data.shape}")
    
    # 检查所有因子数据是否有相同的索引结构
    base_data = factor_data_dict[source_factors[0]]
    
    # 合并所有因子数据到一个DataFrame
    merged_data = base_data[['交易日期', '股票代码']].copy()
    
    # 添加各个因子列
    for factor_name in source_factors:
        factor_df = factor_data_dict[factor_name]
        if factor_name in factor_df.columns:
            merged_data[factor_name] = factor_df[factor_name]
        else:
            print(f"警告: 在 {factor_name} 数据中找不到对应的因子列")
            return False
    
    print(f"合并后数据形状: {merged_data.shape}")
    print(f"包含的因子列: {[col for col in merged_data.columns if col in source_factors]}")
    
    # 应用PCA生成复合因子
    merged_data[target_factor] = compute_pca_by_trade_date(merged_data, source_factors, n_components)
    
    # 创建目标因子目录
    target_factor_path = os.path.join(period_path, target_factor)
    os.makedirs(target_factor_path, exist_ok=True)
    
    # 保存复合因子数据
    output_file = os.path.join(target_factor_path, f'{target_factor}.pkl')
    merged_data.to_pickle(output_file)
    
    print(f"PCA复合因子已保存到: {output_file}")
    print(f"数据形状: {merged_data.shape}")
    print(f"因子 {target_factor} 统计信息:")
    print(merged_data[target_factor].describe())
    
    return True

def create_pca_composite_factor_for_period(period):
    """Backward-compatible alias for legacy scripts."""
    return build_period_pca_factor(period)

def compute_pca_by_trade_date(data, factor_columns, n_components=1):
    """
    按交易日期分组应用PCA
    
    Args:
        data: 包含因子数据的DataFrame
        factor_columns: 要进行PCA的因子列名列表
        n_components: 主成分数量
    
    Returns:
        Series: PCA第一主成分数据
    """
    result_series = pd.Series(index=data.index, dtype=float)
    
    # 按交易日期分组处理
    for date, group in data.groupby('交易日期'):
        # 提取因子数据
        factor_data = group[factor_columns]
        
        # 删除包含NaN的行
        clean_data = factor_data.dropna()
        
        if len(clean_data) < 2:
            # 数据不足时使用简单平均
            if len(clean_data) == 1:
                result_series.loc[group.index] = clean_data.iloc[0].mean()
            else:
                result_series.loc[group.index] = np.nan
            continue
        
        try:
            # 标准化数据
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(clean_data)
            
            # 应用PCA
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(scaled_data)
            
            # 将PCA结果映射回原始索引
            pca_series = pd.Series(pca_result[:, 0], index=clean_data.index)
            result_series.loc[clean_data.index] = pca_series
            
            # 对有NaN的行，使用该日期的PCA均值填充
            nan_mask = group.index.difference(clean_data.index)
            if len(nan_mask) > 0:
                result_series.loc[nan_mask] = pca_series.mean()
                
        except Exception as e:
            print(f"日期 {date} PCA计算出错: {e}")
            # 出错时使用简单平均
            avg_values = clean_data.mean(axis=1)
            result_series.loc[clean_data.index] = avg_values
            
            # 处理NaN行
            nan_mask = group.index.difference(clean_data.index)
            if len(nan_mask) > 0:
                result_series.loc[nan_mask] = avg_values.mean()
    
    return result_series

def apply_pca_by_date(data, factor_columns, n_components=1):
    """Backward-compatible alias for legacy scripts."""
    return compute_pca_by_trade_date(data, factor_columns, n_components)

def main():
    """
    主函数：生成PCA复合因子
    """
    print("=" * 60)
    print("PCA复合因子生成器")
    print("=" * 60)
    print(f"源因子: {', '.join(source_factors)}")
    print(f"目标因子: {target_factor}")
    print(f"处理周期: {', '.join(periods)}")
    print(f"主成分数量: {n_components}")
    print("=" * 60)
    
    # 检查因子数据路径是否存在
    if not os.path.exists(Cfg.factor_path):
        print(f"错误: 因子数据路径不存在 {Cfg.factor_path}")
        print("请先运行 prepare_data.py 生成基础因子数据")
        return False
    
    success_count = 0
    total_count = len(periods)
    
    if multiple_process and len(periods) > 1:
        # 多进程处理
        print("使用多进程模式...")
        results = Parallel(Cfg.n_job)(
            delayed(build_period_pca_factor)(period) 
            for period in periods
        )
        success_count = sum(results)
    else:
        # 单进程处理
        print("使用单进程模式...")
        for period in periods:
            if build_period_pca_factor(period):
                success_count += 1
    
    print("=" * 60)
    print(f"处理完成: {success_count}/{total_count} 个周期成功")
    
    if success_count > 0:
        print(f"\n复合因子 {target_factor} 已生成完成！")
        print("\n下一步操作：")
        print("1. 在策略文件中添加该因子到 factors 字典")
        print("2. 在选股逻辑中使用该复合因子")
        print("3. 运行回测验证因子效果")
        print("\n示例配置：")
        print(f"factors = {{'总市值': [], '{target_factor}': [], ...}}")
    
    return success_count == total_count


if __name__ == "__main__":
    main()






