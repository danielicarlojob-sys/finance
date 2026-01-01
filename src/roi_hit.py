import pandas as pd

def get_first_roi_hit(
    df: pd.DataFrame,
    action: str,
    purchase_date: pd.Timestamp | str,
    roi_target: float = 0.05
) -> dict | None: #tuple[pd.Timestamp, float] | None:
    """
    Returns the first date where the CLOSE price for a given ACTION
    reaches the target ROI relative to the purchase price.

    Args:
        df (pd.DataFrame):
            MultiIndex column DataFrame with levels: ACTION, CURRENCY, METRIC.
        action (str):
            The ACTION identifier to analyze (must exist in df.columns).
        purchase_date (pd.Timestamp | str):
            The assumed purchase date; the first trading date on or after
            this date is used as the entry point.
        roi_target (float):
            Target ROI as a fraction (e.g., 0.05 for 5%).

    Returns:
        tuple[pd.Timestamp, float] | None:
            Returns a tuple (date, close_price) of the first ROI hit,
            or None if target ROI was never achieved.
    """
    if df.columns.nlevels != 3:
        raise ValueError("Expected 3-level MultiIndex columns")

    # Extract sub-DataFrame for the ACTION
    sub = df[action]

    # Drop currency level if present
    if isinstance(sub.columns, pd.MultiIndex):
        sub = sub.droplevel("CURRENCY", axis=1)

    # Ensure CLOSE exists
    if "CLOSE" not in sub.columns:
        raise KeyError(f"{action} missing CLOSE price")

    # Sort and drop empty rows
    sub = sub.sort_index().dropna(subset=["CLOSE"])
    if sub.empty:
        return None

    # Synthesize OPEN if needed (not strictly required for ROI)
    if "OPEN" not in sub.columns:
        sub["OPEN"] = sub["CLOSE"].shift(1)
    if purchase_date is not None:
        purchase_date = pd.Timestamp(purchase_date)

        idx = df.index

        # Align timezone
        if idx.tz is not None and purchase_date.tz is None:
            purchase_date = purchase_date.tz_localize(idx.tz)
        elif idx.tz is None and purchase_date.tz is not None:
            purchase_date = purchase_date.tz_convert(None)
    # Resolve purchase index

    valid_dates = sub.index[sub.index >= purchase_date]
    if valid_dates.empty:
        return None
    purchase_idx = sub.index.get_loc(valid_dates[0])
    buy_price = sub["CLOSE"].iloc[purchase_idx]

    # Compute ROI threshold
    target_price = buy_price * (1 + roi_target)

    # Scan forward for first ROI hit
    for i in range(purchase_idx, len(sub)):
        if sub["CLOSE"].iloc[i] >= target_price:
            target_achieved = {
                'ACTION':action,
                'PURCHASE DATE':purchase_date,
                'PURCHASE PRICE':buy_price,
                'SET ROI TARGET':roi_target,
                'DATE TARGET MET':sub.index[i],
                'EXIT ACTION PRICE':sub["CLOSE"].iloc[i],
                'DATE TO ACHIEVE TARGET':sub.index[i] - purchase_date,
            }
            # return sub.index[i], sub["CLOSE"].iloc[i]
            return target_achieved

    # ROI target never reached
    return None
