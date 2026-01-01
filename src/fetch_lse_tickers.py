import pandas as pd
import requests
from io import StringIO
from src.debug_print import debug_print
from src.utils.retry_decorator import log_exceptions_with_retry

HEADERS = {"User-Agent": "Mozilla/5.0"}

def _read_wiki_table(url: str, ticker_col: str) -> pd.DataFrame:
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()

    tables = pd.read_html(StringIO(resp.text))
    for t in tables:
        if {"Company", ticker_col}.issubset(t.columns):
            return t

    raise RuntimeError(f"No matching table found at {url}")


    raise RuntimeError(f"No matching table found at {url}")

def get_ftse_all_share() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/FTSE_All-Share_Index"

    df = _read_wiki_table(url, ticker_col="EPIC")

    out = (
        df[["Company", "EPIC"]]
        .rename(columns={"EPIC": "LSE_Ticker"})
        .dropna(subset=["LSE_Ticker"])
        .drop_duplicates(subset=["LSE_Ticker"])
        .sort_values("LSE_Ticker")
        .reset_index(drop=True)
    )

    out["Market"] = "FTSE All-Share"
    return out

def get_aim_all_share() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/FTSE_AIM_All-Share_Index"

    df = _read_wiki_table(url, ticker_col="EPIC")

    out = (
        df[["Company", "EPIC"]]
        .rename(columns={"EPIC": "LSE_Ticker"})
        .dropna(subset=["LSE_Ticker"])
        .drop_duplicates(subset=["LSE_Ticker"])
        .sort_values("LSE_Ticker")
        .reset_index(drop=True)
    )

    out["Market"] = "AIM"
    return out



@log_exceptions_with_retry(
    max_retries=5,
    prefix_fn=debug_print,
    retry_delay=1.0,   # optional
)
def get_ftse100():
    url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers).text

    tables = pd.read_html(StringIO(html))
    for table in tables:
        if {"Company", "Ticker"}.issubset(table.columns):
            return table[["Company", "Ticker"]]

    raise RuntimeError("FTSE 100 table not found")

"""
2. Ticker normalisation - For Yahoo Finance compatibility:

    → lse_equities["Yahoo_Ticker"] = lse_equities["LSE_Ticker"] + ".L"

"""

if __name__ == "__main__":
    try:
        ftse100 = get_ftse100()
        ftse100["Yahoo_Ticker"] = ftse100["Ticker"] + ".L"
        print(ftse100)
        l = ftse100["Yahoo_Ticker"].to_list()
        print(len(ftse100))
        print(l)

    except Exception as e:
        print(f"ERROR in ftse100 {type(e).__name__}: {e}")
    
    try:
        ftse_all = get_ftse_all_share()
        print(len(ftse_all), "FTSE All-Share companies")
    except Exception as e:
        print(f"ERROR in ftse100 {type(e).__name__}: {e}")
    
    try:    
        aim = get_aim_all_share()
        print(len(aim), "AIM companies")
    except Exception as e:
        print(f"ERROR in ftse100 {type(e).__name__}: {e}")
    
    try:
        lse_equities = pd.concat([ftse_all, aim], ignore_index=True)
    except Exception as e:
        print(f"ERROR in ftse100 {type(e).__name__}: {e}")
