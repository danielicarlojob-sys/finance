import pandas as pd
import requests
from io import StringIO
from pathlib import Path
import time

HEADERS = {"User-Agent": "Mozilla/5.0"}
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def _cache_request(url: str, name: str) -> str:
    """
    Download a URL to cache, or read from cache if exists.
    """
    file_path = CACHE_DIR / f"{name}.html"
    if file_path.exists():
        return file_path.read_text(encoding="utf-8")
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    file_path.write_text(resp.text, encoding="utf-8")
    time.sleep(0.5)  # polite scraping
    return resp.text

def _get_index_constituents(url: str, ticker_col: str, cache_name: str, source_index: str, market: str) -> pd.DataFrame:
    html = _cache_request(url, cache_name)
    tables = pd.read_html(StringIO(html))
    for t in tables:
        if {"Company", ticker_col}.issubset(t.columns):
            df = t[["Company", ticker_col]].rename(columns={ticker_col: "LSE_Ticker"})
            df["Market"] = market
            df["Source_Index"] = source_index
            # Basic validation
            df = df.dropna(subset=["Company", "LSE_Ticker"]).drop_duplicates(subset="LSE_Ticker")
            return df
    raise RuntimeError(f"No table found at {url}")

# ---- Main Market ----
def get_ftse_all_share() -> pd.DataFrame:
    dfs = [
        _get_index_constituents(
            "https://en.wikipedia.org/wiki/FTSE_100_Index",
            "Ticker",
            "ftse100",
            "FTSE 100",
            "Main"
        ),
        _get_index_constituents(
            "https://en.wikipedia.org/wiki/FTSE_250_Index",
            "Ticker",
            "ftse250",
            "FTSE 250",
            "Main"
        ),
        _get_index_constituents(
            "https://en.wikipedia.org/wiki/FTSE_SmallCap_Index",
            "Ticker",
            "ftse_smallcap",
            "FTSE SmallCap",
            "Main"
        )
    ]
    return pd.concat(dfs, ignore_index=True).sort_values("LSE_Ticker").reset_index(drop=True)

# ---- AIM Market ----
def get_aim_all_share() -> pd.DataFrame:
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    dfs = []
    for letter in letters:
        url = f"https://en.wikipedia.org/wiki/List_of_companies_listed_on_AIM:_{letter}"
        try:
            df = _get_index_constituents(url, "EPIC", f"aim_{letter}", f"AIM {letter}", "AIM")
            dfs.append(df)
        except Exception:
            # Some letters may have no page, skip silently
            continue
    if not dfs:
        raise RuntimeError("No AIM data found")
    return pd.concat(dfs, ignore_index=True).sort_values("LSE_Ticker").reset_index(drop=True)

# ---- Combine ----
def get_all_lse_equities() -> pd.DataFrame:
    main = get_ftse_all_share()
    aim = get_aim_all_share()
    return pd.concat([main, aim], ignore_index=True).sort_values("LSE_Ticker").reset_index(drop=True)

# ---- Export to CSV ----
def save_csv(df: pd.DataFrame, filename: str = "lse_equities.csv"):
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} equities to {filename}")

# ---- Main ----
if __name__ == "__main__":
    print("Building LSE equity universe...")
    lse_equities = get_all_lse_equities()
    save_csv(lse_equities)
