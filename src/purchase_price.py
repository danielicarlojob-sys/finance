import pandas as pd
from src.debug_print import debug_print

def get_purchase_price(
    df: pd.DataFrame,
    action: str,
    currency: str,
    metric: str,
    date: pd.Timestamp,
) -> float | None:
    """
    Retrieve the purchase price for a given action, currency, and date
    from a MultiIndex DataFrame.

    Parameters:
    - df: pd.DataFrame
        DataFrame with MultiIndex columns (ACTION, CURRENCY, METRIC)
    - action: str
        The action identifier (e.g., stock ticker)
    - currency: str
        The currency identifier (e.g., 'USD', 'GBP')
    - metric: str
        The metric to retrieve (e.g., 'LOW', 'HIGH', 'CLOSE')
    - date: pd.Timestamp
        The date for which to retrieve the purchase price

    Returns:
    - float | None
        The purchase price if found, otherwise None
    """
    try:
        series = df[(action, currency, metric)]

        value = series.loc[:date].iloc[-1]
        return value    
    except KeyError as e:
        debug_print(f"{debug_print()}\nKeyError retrieving purchase price for {action} in {currency} on {date} {type(e).__name__}: {e}")
        value = None
        return value