import pandas as pd
import numpy as np

def detect_undervalued(df: pd.DataFrame, pe_threshold: float = 15, pb_threshold: float = 1.5):
    """
    Detect undervalued shares based on P/E and P/B ratios.

    Args:
        df (pd.DataFrame): Must have columns ['ACTION', 'Price', 'EPS', 'BookValue']
        pe_threshold (float): Max P/E ratio to consider undervalued
        pb_threshold (float): Max P/B ratio to consider undervalued

    Returns:
        pd.DataFrame: Filtered and sorted shares by attractiveness
    """
    df = df.copy()

    # Compute ratios
    df['P/E'] = df['Price'] / df['EPS']
    df['P/B'] = df['Price'] / df['BookValue']

    # Identify undervalued shares
    df['UndervaluedScore'] = 0
    df.loc[df['P/E'] < pe_threshold, 'UndervaluedScore'] += 1
    df.loc[df['P/B'] < pb_threshold, 'UndervaluedScore'] += 1

    # Sort by UndervaluedScore, then lowest P/E, then lowest P/B
    df_sorted = df.sort_values(
        by=['UndervaluedScore', 'P/E', 'P/B'],
        ascending=[False, True, True]
    )

    return df_sorted

if __name__ == "__main__":
    # Example usage:
    data = pd.DataFrame({
        'ACTION': ['AAA', 'BBB', 'CCC', 'DDD'],
        'Price': [100, 50, 200, 30],
        'EPS': [10, 5, 25, 2],
        'BookValue': [90, 40, 180, 20]
    })

    undervalued_shares = detect_undervalued(data)
    print(undervalued_shares)
