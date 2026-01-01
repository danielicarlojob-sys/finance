import pandas as pd

def extract_latest_fundamentals(
    df: pd.DataFrame,
    evaluation_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Reduce ACTION/CURRENCY/METRIC DataFrame to one row per ACTION
    using either:
      - the latest available data, or
      - the latest data on or before `evaluation_date` if provided.

    Parameters
    ----------
    df : pd.DataFrame
        MultiIndex DataFrame with columns (ACTION, CURRENCY, METRIC)
        and a DatetimeIndex.
    evaluation_date : pd.Timestamp | None
        Optional evaluation date. If provided, fundamentals are
        extracted from the most recent trading day <= this date.

    Returns
    -------
    pd.DataFrame
        Index: ACTION
        Columns: Price, EPS, BookValue, Dividend
    """

    actions = df.columns.get_level_values("ACTION").unique()
    rows = []

    for action in actions:
        sub = df[action]

        # Drop currency level
        if isinstance(sub.columns, pd.MultiIndex):
            sub = sub.droplevel("CURRENCY", axis=1)

        # Remove fully empty rows
        sub = sub.dropna(how="all")
        if sub.empty:
            continue

        # -------------------------------------------------
        # RESOLVE EVALUATION ROW  (← THIS IS THE FIX)
        # -------------------------------------------------
        if evaluation_date is not None:
            evaluation_date = pd.Timestamp(evaluation_date)

            # Align timezone between index and evaluation_date
            if sub.index.tz is not None and evaluation_date.tz is None:
                evaluation_date = evaluation_date.tz_localize(sub.index.tz)
            elif sub.index.tz is None and evaluation_date.tz is not None:
                evaluation_date = evaluation_date.tz_convert(None)

            valid_dates = sub.index[sub.index <= evaluation_date]
            if valid_dates.empty:
                # No data available before evaluation_date
                continue

            row = sub.loc[valid_dates[-1]]
        else:
            row = sub.iloc[-1]

        # -------------------------------------------------
        # EXTRACT FUNDAMENTALS + PRICE
        # -------------------------------------------------
        rows.append({
            "ACTION": action,
            "Price": row["CLOSE"],
            "EPS": row.get("EPS"),
            "BookValue": row.get("BookValue"),
            "Dividend": row.get("Dividend"),
        })

    return pd.DataFrame(rows).set_index("ACTION")
