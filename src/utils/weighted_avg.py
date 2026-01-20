from re import U
import pandas as pd
import numpy as np
from typing import Tuple

def weighted_avg(
    df_portfolio: pd.DataFrame,
    df_input_share: pd.DataFrame,
    share_price = "Price / share",
    share_count = "No. of shares",
    time_col = "Time",
    ID_col = "ID",
) -> Tuple[float, pd.Timestamp, float, list]:

    # --- validate required columns ---
    required_cols = {share_price, share_count, time_col, ID_col}
    for df, name in [(df_portfolio, "df_portfolio"), (df_input_share, "df_input_share")]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")

    # --- total shares ---
    total_shares = (
        df_portfolio[share_count].sum()
        + df_input_share[share_count].sum()
    )

    if total_shares == 0:
        raise ValueError("Total number of shares is zero")

    # --- weighted average price ---
    weighted_price = (
        (df_portfolio[share_price] * df_portfolio[share_count]).sum()
        + (df_input_share[share_price] * df_input_share[share_count]).sum()
    ) / total_shares

    # --- weighted average datetime (convert to int64 ns) ---
    portfolio_time_ns = df_portfolio[time_col].astype("int64")
    input_time_ns = df_input_share[time_col].astype("int64")

    weighted_time_ns = (
        (portfolio_time_ns * df_portfolio[share_count]).sum()
        + (input_time_ns * df_input_share[share_count]).sum()
    ) / total_shares

    weighted_time = pd.to_datetime(int(weighted_time_ns))
    df_portfolio[ID_col]
    idx_portfolio = df_portfolio.index.tolist()[0]
    idx_input_share = df_input_share.index.tolist()[0]
    print(f"idx_portfolio: {idx_portfolio}, idx_input_share: {idx_input_share}")
    if isinstance(df_portfolio.at[idx_portfolio, ID_col], list):
        updated_ID_list = df_portfolio.at[idx_portfolio, ID_col] + [df_input_share.at[idx_input_share, ID_col]]
    else:
        updated_ID_list = [df_portfolio.at[idx_portfolio, ID_col], df_input_share.at[idx_input_share, ID_col]]

    return weighted_price, weighted_time, updated_ID_list

import pandas as pd
from typing import Tuple, Set

def weighted_avg_v2(
    df_portfolio: pd.DataFrame,
    df_input_share: pd.DataFrame,
    share_price: str = "Price / share",
    share_count: str = "No. of shares",
    time_col: str = "Time",
    ID_col: str = "ID",
) -> Tuple[float, pd.Timestamp, Set[str]]:

    # --- validate required columns ---
    required_cols = {share_price, share_count, time_col, ID_col}
    for df, name in ((df_portfolio, "df_portfolio"), (df_input_share, "df_input_share")):
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")

    # --- total shares ---
    total_shares = (
        df_portfolio[share_count].sum()
        + df_input_share[share_count].sum()
    )

    if total_shares == 0:
        raise ValueError("Total number of shares is zero")

    # --- weighted average price ---
    weighted_price = (
        (df_portfolio[share_price] * df_portfolio[share_count]).sum()
        + (df_input_share[share_price] * df_input_share[share_count]).sum()
    ) / total_shares

    # --- weighted average datetime (ns since epoch) ---
    portfolio_time_ns = df_portfolio[time_col].astype("int64")
    input_time_ns = df_input_share[time_col].astype("int64")

    weighted_time_ns = (
        (portfolio_time_ns * df_portfolio[share_count]).sum()
        + (input_time_ns * df_input_share[share_count]).sum()
    ) / total_shares

    weighted_time = pd.to_datetime(int(weighted_time_ns))

    # --- merge IDs into a set ---
    idx_portfolio = df_portfolio.index[0]
    idx_input_share = df_input_share.index[0]

    def to_set(x) -> set:
        if isinstance(x, set):
            return x
        if isinstance(x, list):
            return set(x)
        return {x}

    updated_ID_set = (
        to_set(df_portfolio.at[idx_portfolio, ID_col])
        | to_set(df_input_share.at[idx_input_share, ID_col])
    )

    return weighted_price, weighted_time, total_shares,updated_ID_set
