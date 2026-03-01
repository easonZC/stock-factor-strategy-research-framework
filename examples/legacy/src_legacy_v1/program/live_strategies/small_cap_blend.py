"""
Live strategy module: small_cap_blend.
"""

import pandas as pd

name = file_name = __file__.replace("\\", "/").split("/")[-1].replace(".py", "")
period_offset = ["W_0"]
factors = {"总市值": [], "布林带位置": [], "风格因子": []}
select_count = 3


def screen_universe(data: pd.DataFrame) -> pd.DataFrame:
    """Apply listing/liquidity/tradability filters before ranking."""
    data = data[data["交易日期"] >= "20070101"]
    data = data[
        ~((data["股票代码"] == "sz300156") & (data["交易日期"] >= pd.to_datetime("2020-04-10")))
    ]
    data = data[
        ~((data["股票代码"] == "sz300362") & (data["交易日期"] >= pd.to_datetime("2020-04-10")))
    ]
    data = data[data["股票名称"].str.contains("ST") == False]
    data = data[data["股票名称"].str.contains("S") == False]
    data = data[data["股票名称"].str.contains("\\*") == False]
    data = data[data["股票名称"].str.contains("退") == False]
    data = data[data["交易天数"] / data["市场交易天数"] >= 0.8]
    data = data[data["下日_是否交易"] == 1]
    data = data[data["下日_开盘涨停"] == False]
    data = data[data["下日_是否ST"] == False]
    data = data[data["下日_是否退市"] == False]
    data = data[data["上市至今交易天数"] > 250]
    return data


def rank_and_pick(data: pd.DataFrame, count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build score column and select top-N names each rebalance date."""
    data["总市值_排名"] = data.groupby("交易日期")["总市值"].rank(ascending=True)
    data["复合因子"] = data["BB_position"]
    data.dropna(subset=["复合因子"], inplace=True)
    data = data[data["交易日期"] >= pd.to_datetime("2007-07-01")]
    grouped_view = data.copy()
    data["复合因子_排名"] = data.groupby("交易日期")["复合因子"].rank(ascending=False)
    data = data[data["复合因子_排名"] <= count]
    data["选股排名"] = data["复合因子_排名"]
    return data, grouped_view


def filter_stock(all_data: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper expected by the framework."""
    return screen_universe(all_data)


def select_stock(all_data: pd.DataFrame, count: int, params=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compatibility wrapper expected by the framework."""
    _ = params
    return rank_and_pick(all_data, count)



