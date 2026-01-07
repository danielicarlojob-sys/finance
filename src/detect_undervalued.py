import os
import pandas as pd
import numpy as np
from datetime import datetime
from src.fetch_lse_tickers import get_ftse100
from src.exchange_rates_v2 import get_share_prices_2_with_fundamentals
from src.extract_latest_fundamentals import extract_latest_fundamentals
from src.utils.email_sender import send_email_html_multi_inline_images

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
    df['P/E'] = np.where(df['EPS'] > 0, df['Price'] / df['EPS'], np.nan)
    df['P/B'] = np.where(df['BookValue'] > 0, df['Price'] / df['BookValue'], np.nan)


    # Identify undervalued shares
    df['UndervaluedScore'] = 0
    df.loc[df['P/E'] < pe_threshold, 'UndervaluedScore'] += 1
    df.loc[df['P/B'] < pb_threshold, 'UndervaluedScore'] += 1

    # Sort by UndervaluedScore, then lowest P/E, then lowest P/B
    df_sorted = df.sort_values(
        by=['UndervaluedScore', 'P/E', 'P/B', 'Dividend'],
        ascending=[False, True, True, False]
    )

    return df_sorted

if __name__ == "__main__":
    from dotenv import load_dotenv


    load_dotenv()
    email_sender = os.getenv("EMAIL_SENDER")
    email_sender_psw = os.getenv("EMAIL_SENDER_PSW")
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(months=12)

    purchase_date = pd.Timestamp(2026,1,2)
    ROI_target = 0.135


    ftse100 = get_ftse100()
    print(f"ftse100:\n{ftse100}")
    try:
        shares_lse = ftse100["Ticker"].to_list()
    except Exception as e:
        print(f"ERROR in shares_lse: {type(e).__name__}: {e}")


    print("►►► START `get_share_prices_2_with_fundamentals`")
    df_shares_fund, failed_tickers_list  = get_share_prices_2_with_fundamentals(
    tickers=[s+'.L' if not s.endswith('.L') else s for s in shares_lse],
    start=start_date,
    end=end_date,
    base_currency = 'GBP',
    vol_window = 20,
    )
    print("►►► END `get_share_prices_2_with_fundamentals`")

    df_fund= extract_latest_fundamentals(
    df=df_shares_fund,
    evaluation_date=purchase_date,  
    )


    undervalued_shares = detect_undervalued(df_fund)
    filt = undervalued_shares[undervalued_shares["UndervaluedScore"] > 0]


    undervalued_shares_list = filt.index.to_list()

    print(f"→→→ undervalued_shares_list:\n{undervalued_shares_list}")


    actions = [act.split('_')[0] for act in filt.index] 
    action_original_currency = [act.split('_')[1].split('→')[0] for act in filt.index] 
    action_target_currency = [act.split('_')[1].split('→')[1] for act in filt.index] 
    companies = ftse100[ftse100["Ticker"].isin(actions)]


    filt = filt.copy()
    filt["Ticker"] = filt.index.str.split(".").str[0]

    filt["Company"] = (
        filt["Ticker"]
        .map(ftse100.set_index("Ticker")["Company"])
    )
    currency_convertion = [s.split('_')[1] for s in filt.index.to_list()]
    print(f"currency_convertion:{currency_convertion}")

    filt['original currency'] = [s.split('→')[0] for s in currency_convertion]
    filt['converted currency'] = [s.split('→')[1] for s in currency_convertion]
    filt = filt[['Company', 'Ticker', 'UndervaluedScore', 'original currency', 'converted currency', 
                 'Price', 'EPS', 'BookValue', 'Dividend', 'P/E', 'P/B']]

    print(f"→→→ filt:\n{filt}")
    from src.plot_shares_ROI import plot_candles_volatility_volume_roi as ROI

    ROI(
    df=df_shares_fund,
    actions=filt.index.to_list(),
    start=df_shares_fund.index.min(),
    end=df_shares_fund.index.max(),
    purchase_date= purchase_date,
    roi_target=ROI_target
)
    filt.to_csv("filt.csv", index=False)
    from src.utils.email_undervalued_shares import build_undervalued_shares_email_with_images

    try:
        text_body, html_body, inline_images = build_undervalued_shares_email_with_images(
            df_undervalued_shares=filt,
            image_dir='output'
        )
        print(f"text_body:\n{text_body}")
        print(f"html_body:\n{html_body}")
        print(f"inline_images:\n{inline_images}")
    except Exception as e:
        print(f"ERROR in build_undervalued_shares_email_with_images {type(e).__name__}: {e}")
    try:
        send_email_html_multi_inline_images(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            username=email_sender,
            password=email_sender_psw,
            sender=email_sender,
            recipients=["ingcarldan@gmail.com"],
            subject=f"FTSE100 - Undevalued shares – {datetime.today():%d-%m-%Y %H:%M}",
            text_body=text_body,
            html_body=html_body,
            inline_images=inline_images,
        )
    except Exception as e:
        print(f"Could not run send_email_html_multi_inline_images {type(e).__name__}: {e}")
        


