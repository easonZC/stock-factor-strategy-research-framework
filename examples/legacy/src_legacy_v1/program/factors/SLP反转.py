"""
Factor definition module: SLP反转.
"""

name = file_name = __file__.replace('\\', '/').split('/')[-1].replace('.py', '') 

ipt_fin_cols = []  # 输入的财务字段
opt_fin_cols = []  # 输出的财务字段

# 新增拐点识别所需的数据结构
def special_data():
    '''
    初始化拐点数据结构
    每只股票维护：最近拐点索引、待确认拐点索引、价格序列
    '''
    return {
        'last_pivot_index': {},  # 股票代码: 最近确认拐点的索引
        'candidate_index': {},   # 股票代码: 待确认拐点索引
        'price_series': {}       # 股票代码: [价格序列]
    }

def after_resample(data):
    return data

def cal_cross_factor(data):
    return data

def cal_factors(data, fin_data, fin_raw_data, exg_dict):
    """
    计算SLP动态拐点反转因子
    1. 使用动态拐点替代固定时间窗口
    2. 基于价格分段识别反转形态
    
    :param data: 包含股票数据的DataFrame
    :param exg_dict: 因子扩展字典
    :return: 包含SLP因子的DataFrame
    """
    DELTA_P = 1.0  # 拐点确认阈值（文档未明确，初步尝试）

    # 初始化特殊数据结构
    if not hasattr(data, 'special_data'):
        data.special_data = special_data()
    
    # 存储每只股票的价格序列
    for symbol, group in data.groupby('股票代码'):
        if symbol not in data.special_data['price_series']:
            data.special_data['price_series'][symbol] = group['收盘价'].tolist()
    
    data = data.reset_index(drop=True)

    # 为每只股票初始化状态
    special_state = special_data()
    
    def process_group(group):
        symbol = group['股票代码'].iloc[0]
        group = group.reset_index(drop=True)
        
        # 存储价格序列
        special_state['price_series'][symbol] = group['收盘价'].tolist()
        
        # 计算每行的SLP因子值
        group['SLP反转'] = group.apply(
            lambda row: calculate_slp(
                symbol=symbol,
                current_index=row.name,
                special_data=special_state,
                delta_p=DELTA_P
            ), axis=1
        )
        return group

    data = data.groupby('股票代码', group_keys=False).apply(process_group)

    exg_dict['SLP反转'] = 'last'
    
    return data, exg_dict

def calculate_slp(symbol, current_index, special_data, delta_p):
    """
    参数：
    symbol - 股票代码
    current_index - 当前数据点在序列中的位置
    special_data - 存储拐点状态的数据结构
    delta_p - 拐点确认阈值
    """
    prices = special_data['price_series'].get(symbol, [])
    
    # 检查数据是否足够
    if current_index < 1 or len(prices) < 2:
        return None
    
    # 初始化拐点状态
    if symbol not in special_data['last_pivot_index']:
        special_data['last_pivot_index'][symbol] = 0  # 第一个点作为初始拐点
    
    if symbol not in special_data['candidate_index']:
        special_data['candidate_index'][symbol] = None
    
    # 获取当前拐点状态
    last_pivot_idx = special_data['last_pivot_index'][symbol]
    candidate_idx = special_data['candidate_index'][symbol]
    
    # 1. 检查是否需要设置新候选拐点
    if candidate_idx is None:
        # 将当前点设为候选拐点（文档中算法：从最近拐点后寻找待确认拐点）
        special_data['candidate_index'][symbol] = current_index
        candidate_idx = current_index
    
    # 2. 尝试确认候选拐点
    price_last_pivot = prices[last_pivot_idx]
    price_candidate = prices[candidate_idx]
    
    # 计算AB段收益率（最近拐点到候选点）
    r_ab = (price_candidate - price_last_pivot) / price_last_pivot
    
    # 计算BC段收益率（候选点到当前点）
    r_bc = (prices[current_index] - price_candidate) / price_candidate
    
    # 3. 检查是否满足拐点确认条件
    if abs(r_bc) > abs(r_ab) * delta_p:
        # 确认候选点为有效拐点
        special_data['last_pivot_index'][symbol] = candidate_idx
        special_data['candidate_index'][symbol] = None
    
    # 4. 计算SLP因子值
    last_pivot_idx = special_data['last_pivot_index'][symbol]
    price_pivot = prices[last_pivot_idx]
    price_current = prices[current_index]
    
    # 时间间隔
    delta_t = current_index - last_pivot_idx
    
    # 避免除零错误
    if delta_t == 0 or price_pivot == 0:
        return 0.0

    slp = (price_current - price_pivot) / (price_pivot * delta_t)
    
    return slp


