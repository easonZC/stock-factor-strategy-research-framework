"""
Research strategy: keep names starting with "中", then select smallest market-cap stocks.
"""

import pandas as pd

name = file_name = __file__.replace("\\", "/").split("/")[-1].replace(".py", "")
period_offset = ["3_0"]
factors = {"总市值": []}
select_count = 3


def screen_universe(data: pd.DataFrame) -> pd.DataFrame:
    """Apply baseline tradability and risk filters."""
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
    data = data[data["股票名称"].str.startswith("中") == True]
    return data


def rank_and_pick(data: pd.DataFrame, count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank by ascending market cap and keep top-N per date."""
    data["总市值_排名"] = data.groupby("交易日期")["总市值"].rank(ascending=True)
    data["复合因子"] = data["总市值_排名"]
    data.dropna(subset=["复合因子"], inplace=True)
    data = data[data["交易日期"] >= pd.to_datetime("2009-01-01")]
    grouped_view = data.copy()
    data["复合因子_排名"] = data.groupby("交易日期")["复合因子"].rank(ascending=True)
    data = data[data["复合因子_排名"] <= count]
    data["选股排名"] = data["复合因子_排名"]
    return data, grouped_view


def drawdown_switch_signal(equity_df: pd.DataFrame, window: int, max_dd: float) -> pd.DataFrame:
    """Generate periodic on/off signal by rolling drawdown."""
    df = equity_df.copy()
    df["max_equity"] = df["equity_curve"].rolling(window, min_periods=1).max()
    df["mdd"] = 1 - df["equity_curve"] / df["max_equity"]
    df.loc[df["mdd"] < max_dd, "signal"] = 1
    df["signal"].fillna(value=0, inplace=True)
    period_df = df.groupby(["周期"]).agg({"交易日期": "last", "signal": "last"})
    return period_df


def filter_stock(all_data: pd.DataFrame) -> pd.DataFrame:
    """Framework compatibility wrapper."""
    return screen_universe(all_data)


def select_stock(all_data: pd.DataFrame, count: int, params=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Framework compatibility wrapper."""
    _ = params
    return rank_and_pick(all_data, count)


def timing(equity: pd.DataFrame, params=None) -> pd.DataFrame:
    """Framework timing hook."""
    _ = params
    return drawdown_switch_signal(equity, 10, 0.10)




