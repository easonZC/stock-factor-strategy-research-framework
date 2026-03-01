"""
Global runtime settings for the stock-selection framework.
"""

import os

# ===== Runtime window =====
date_start = "2007-07-01"
date_end = None

# ===== Trading controls =====
buy_method = "开盘"
run_all_offset = True
subset_num = None

# ===== Strategy selection =====
stg_file = "small_cap_blend"
stg_folder = "live_strategies"  # live_strategies / research_strategies
other_factors = {}

# ===== Execution mode =====
end_exchange = False

# ===== Path setup =====
_current_dir = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.abspath(os.path.join(_current_dir, "..", ".."))

# User-provided datasets
stock_data_path = os.path.join(root_path, "data", "stock_data")
fin_path = None

# Built-in reference data
index_path = os.path.join(root_path, "sh000001.csv")
period_offset_file = os.path.join(root_path, "period_offset.csv")

# Output data
factor_path = os.path.join(root_path, "data", "因子数据")

# Costs
c_rate = 1.2 / 10000
t_rate = 1 / 1000

# Threading
n_job = min(max((os.cpu_count() or 2) - 1, 1), 60)

if end_exchange:
    buy_method = None



