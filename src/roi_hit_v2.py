import pandas as pd

def get_first_roi_hit(
    df: pd.DataFrame,
    action: str,
    purchase_date: pd.Timestamp | str,
    roi_target: float = 0.05
) -> dict: #tuple[pd.Timestamp, float] | None:
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
                'PURCHASE DATE':purchase_date.strftime("%d/%m/%Y"),
                'PURCHASE PRICE': round(buy_price, 2),
                'SET ROI TARGET':roi_target,
                'DATE TARGET MET':sub.index[i].strftime("%d/%m/%Y"),
                'EXIT ACTION PRICE': round(sub["CLOSE"].iloc[i], 2),
                'DAYS TO ACHIEVE TARGET':(sub.index[i] - purchase_date).days,
            }
            # return sub.index[i], sub["CLOSE"].iloc[i]
            return target_achieved

    # ROI target never reached

    # ---- fallback: ROI target never met ----
    last_idx = sub.index[-1]
    last_close = sub["CLOSE"].iloc[-1]

    return {
        'ACTION': action,
        'PURCHASE DATE': purchase_date.strftime("%d/%m/%Y"),
        'PURCHASE PRICE': round(buy_price, 2),
        'SET ROI TARGET': roi_target,
        'DATE TARGET MET': None,
        'EXIT ACTION PRICE': round(last_close, 2),
        'DAYS TO ACHIEVE TARGET': None,
}





if __name__ == "__main__":
    from pprint import pprint as pp
    shares = ['FRES.L_GBP→GBP', 'ENT.L_GBP→GBP', 
              'GLEN.L_GBP→GBP', 'MNG.L_GBP→GBP', 
              'PHNX.L_GBP→GBP', 'VOD.L_GBP→GBP']
    # df = pd.read_csv("df_shares_fund.csv", index_col=[0, 1])
    df = pd.read_pickle("df_shares_fund.pkl")

    print(df)
    ROI = {}
    for share in shares:
        share_clear = share.split('.')[0]
        ROI[share_clear] = get_first_roi_hit(
            df=df,
            action=share,
            purchase_date=pd.Timestamp(2026,1,2),
            roi_target= 0.05
        )
    pp(ROI, indent=4)