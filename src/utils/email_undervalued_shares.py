import os
import pandas as pd
from datetime import datetime        # Datetime handling
from typing import Iterable, Optional
import smtplib
from email.message import EmailMessage
from pathlib import Path
import mimetypes
from src.debug_print import debug_print
from src.utils.retry_decorator import log_exceptions_with_retry
from dotenv import load_dotenv


load_dotenv()
email_sender = os.getenv("EMAIL_SENDER")
email_sender_psw = os.getenv("EMAIL_SENDER_PSW")

def build_undervalued_shares_email_with_images(
    df_undervalued_shares: pd.DataFrame,
    image_dir: Path | str | None = None,
    image_map: dict[str, Path | str] | None = None,
):
    """
    Build plain-text and HTML email bodies for ROI notifications, including
    optional per-action inline images.

    Args:
        df_undervalued_shares (dict):
            Mapping of action ticker → ROI metadata dictionary.
            Required fields per action:
              - ACTION (str): action identifier including currency (e.g. "AAL.L_GBP→GBP")

        image_dir (Path | str | None):
            Optional directory containing per-action images named
            `{ACTION}_ROI.png`. Used only if `image_map` is not provided
            for a given action.

        image_map (dict[str, Path | str] | None):
            Optional explicit mapping:
              - key (str): action ticker
              - value (Path | str): image file path
            Overrides `image_dir` on a per-action basis.

    Returns:
        tuple:
            text_body (str):
                Plain-text email body used as fallback for non-HTML clients.

            html_body (str):
                HTML email body containing:
                  - summary table of ROI data
                  - inline image sections referenced via CID.

            inline_images (dict[str, Path]):
                Mapping of content-id → image path.
                Used by the email sender to attach inline images.

    Notes:
        - Missing images are silently skipped.
        - No files are opened; only paths are resolved.
        - Function is side-effect free and safe to call inside ROI loops.
    """
    image_dir = os.path.join(os.getcwd(), image_dir) if image_dir else None
    print(f"{debug_print()} image_dir: {image_dir}")
   
    now = datetime.today().strftime("%d-%m-%Y %H:%M")
    text_lines = [f"Undervalud shares analysis performed on {now}\n"]
    table_rows = []
    image_sections = []
    inline_images = {}

    try:
        png_files_list = [f for f in os.listdir(image_dir) if f.endswith(".png")]
    except Exception as e:
        print(f"{debug_print()} ERROR png_files_list {type(e).__name__}: {e}")
    
    for action in df_undervalued_shares.index.to_list():

        share_name = df_undervalued_shares.loc[action, 'Company']
        share_ticker = df_undervalued_shares.loc[action, 'Ticker']

        currency = df_undervalued_shares.loc[action, 'converted currency']
        cid = f"{action}_img"

        # --- TEXT BODY ---
        text_lines.extend([
            f"Action: {share_ticker}",
            f"Company: {share_name}",
            f"  UndervaluedScore: {df_undervalued_shares.loc[action, 'UndervaluedScore']}",
            f"  Converted Share Price to base currency: {df_undervalued_shares.loc[action, 'Price']:.2f} {currency}",
            f"  Earining per share EPS: {df_undervalued_shares.loc[action, 'EPS']:.2f}",
            f"  BookValue: {df_undervalued_shares.loc[action, 'BookValue']:.4f}",
            f"  Dividend: {df_undervalued_shares.loc[action, 'Dividend']:.4f}",
            f"  P/E: {df_undervalued_shares.loc[action, 'P/E']:.4f}",
            f"  P/B: {df_undervalued_shares.loc[action, 'P/B']:.4f}",
            "",
        ])

        # --- TABLE ---
        table_rows.append(f"""
        <tr>
            <td><b>{share_ticker}</b></td>
            <td><b>{share_name}</b></td>
            <td>{df_undervalued_shares.loc[action, 'UndervaluedScore']}</td>
            <td>{df_undervalued_shares.loc[action, 'Price']:.2f} {currency}</td>
            <td>{df_undervalued_shares.loc[action, 'EPS']:.2f}</td>
            <td>{df_undervalued_shares.loc[action, 'BookValue']:.4f}</td>
            <td>{df_undervalued_shares.loc[action, 'Dividend']:.4f}</td>
            <td>{df_undervalued_shares.loc[action, 'P/E']:.4f}</td>
            <td>{df_undervalued_shares.loc[action, 'P/B']:.4f}</td>
        </tr>
        """)

        # --- IMAGE RESOLUTION ---
        img_path = None
        if image_map and action in image_map:
            img_path = image_map[action]
        elif image_dir:
            try:
                image_action_name = [s for s in png_files_list if share_ticker in s][0]
            except Exception as e:
                print(f"{debug_print()} ERROR in image_action_name extraction")
                image_action_name = share_ticker+".L"+str(df_undervalued_shares.loc[action, 'original currency'])+"→"+str(df_undervalued_shares.loc[action, 'converted currency'])+".png"
            try:
                candidate = os.path.join(image_dir, image_action_name)
            except Exception as e:
                print(f"{debug_print()} candpidate error {type(e).__name__}: {e}")

            if Path(candidate).exists():
                img_path = candidate

        if img_path:
            inline_images[cid] = img_path
            image_sections.append(f"""
            <h4>{action} – Price evolution</h4>
            <img src="cid:{cid}" style="max-width:900px; margin-bottom:20px;">
            """)

        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2>Undervalued Shares</h2>
            <p>Data extracted on <b>{now}</b></p>

            <table border="1" cellpadding="6" cellspacing="0">
                <tr style="background:#f0f0f0;">
                    <th>Action code</th>
                    <th>Company name</th>
                    <th>Undervalued Score</th>
                    <th>Price</th>
                    <th>EPS</th>
                    <th>BookValue</th>
                    <th>Dividend</th>
                    <th>P/E</th>
                    <th>P/B</th>
                </tr>
                {''.join(table_rows)}
            </table>

            <hr>
            {''.join(image_sections)}
        </body>
        </html>
        """
 
    return "\n".join(text_lines), html_body, inline_images
# ==========================
# SCRIPT ENTRY POINT FOR MANUAL TEST
# ==========================
if __name__ == "__main__":
    import os
    root_dir = os.getcwd()
    
    filt = pd.read_csv("filt.csv")
    print(filt)

    try:
        text_body, html_body, inline_images = build_undervalued_shares_email_with_images(
            df_undervalued_shares=filt,
            image_dir=os.path.join(root_dir,'output')
        )
        print(f"text_body:\n{text_body}")
        print(f"html_body:\n{html_body}")
        print(f"inline_images:\n{inline_images}")
    except Exception as e:
        print(f"{debug_print()} ERROR in build_undervalued_shares_email_with_images {type(e).__name__}: {e}")

    