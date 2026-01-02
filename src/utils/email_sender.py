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
# ==================================
# HELPER FUNCTION
# ==================================

def email_text_body_single_action(data:pd.DataFrame)->str: 

    currency = data['ACTION'].split("→")[1]
    body += f"\n    Action: {data['ACTION']} - target ROI:{data['SET ROI TARGET']*100}%"
    body += f"\n        Purchase Date: {data['PURCHASE DATE'].strftime("%d-%m-%Y %H:%M")}"
    body += f"\n        Purchase Price: {round(data['PURCHASE PRICE'], 2)} {currency}"
    body += f"\n        Suggested Sell Date: {data['DATE TARGET MET'].strftime("%d-%m-%Y %H:%M")}"
    body += f"\n        Suggested Sell Price: {round(data['EXIT ACTION PRICE'], 2)} {currency}"
    body += f"\n        Suggested Sell Date: {data['DATE TARGET MET'].strftime("%d-%m-%Y %H:%M")}"           
    body += f"\n        Days to achieve set ROI: {data['DATE TO ACHIEVE TARGET']}\n\n"

    return body

def build_html_body(report_text: str) -> str:
    # Escape basic HTML chars if needed, then replace newlines
    escaped = (report_text
               .replace("&", "&amp;")
               .replace("<", "&lt;")
               .replace(">", "&gt;"))
    html_content = escaped.replace("\n", "<br>\n")
    HTML_body = f"""\
<html>
  <body>
    <p>Data extracted report:</p>
    <div style="font-family: Arial, sans-serif; white-space: pre-wrap;">
      {html_content}
    </div>
  </body>
</html>
"""
    return HTML_body

def build_roi_email_content(roi_data: dict) -> tuple[str, str]:
    """
    Returns (text_body, html_body)
    """

    now = datetime.today().strftime("%d-%m-%Y %H:%M")

    text_lines = [f"Data extracted on {now}\n"]
    html_rows = []

    for ac, data in roi_data.items():
        currency = data["ACTION"].split("→")[1]

        text_lines.extend([
            f"Action: {ac}",
            f"  Target ROI: {data['SET ROI TARGET'] * 100:.1f}%",
            f"  Purchase date: {data['PURCHASE DATE'].strftime('%d-%m-%Y %H:%M')}",
            f"  Purchase price: {data['PURCHASE PRICE']:.2f} {currency}",
            f"  Sell date: {data['DATE TARGET MET'].strftime('%d-%m-%Y %H:%M')}",
            f"  Sell price: {data['EXIT ACTION PRICE']:.2f} {currency}",
            f"  Time to target: {data['DATE TO ACHIEVE TARGET']}",
            "",
        ])

        html_rows.append(f"""
        <tr>
            <td><b>{ac}</b></td>
            <td>{data['SET ROI TARGET'] * 100:.1f}%</td>
            <td>{data['PURCHASE PRICE']:.2f} {currency}</td>
            <td>{data['PURCHASE DATE'].strftime('%d-%m-%Y %H:%M')}</td>
            <td>{data['EXIT ACTION PRICE']:.2f} {currency}</td>
            <td>{data['DATE TARGET MET'].strftime('%d-%m-%Y %H:%M')}</td>
            <td>{data['DATE TO ACHIEVE TARGET']}</td>
        </tr>
        """)

    text_body = "\n".join(text_lines)

    html_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif;">
        <h2>ROI Target Reached</h2>
        <p>Data extracted on <b>{now}</b></p>

        <table border="1" cellpadding="6" cellspacing="0">
            <tr style="background:#f0f0f0;">
                <th>Action</th>
                <th>Target ROI</th>
                <th>Buy Price</th>
                <th>Buy Date</th>
                <th>Sell Price</th>
                <th>Sell Date</th>
                <th>Time to Target</th>
            </tr>
            {''.join(html_rows)}
        </table>

        <br>
        <p><b>Price evolution:</b></p>
        <img src="cid:image1" style="max-width:800px;">
    </body>
    </html>
    """

    return text_body, html_body


# ==================================
# MAIN FUNCTION
# ==================================


def send_email_with_image(
    *,
    smtp_server: str,
    smtp_port: int,
    username: str,
    password: str,
    sender: str,
    recipients: list[str],
    subject: str,
    body: str,
    image_path: str | Path,
):
    """
    Send an email with an image attachment.

    Parameters
    ----------
     - smtp_server : str
        SMTP server hostname (e.g. "smtp.gmail.com")
     - smtp_port : int
        SMTP port (usually 587 for TLS)
     - username : str
        SMTP username (often same as sender email)
     - password : str
        SMTP password or app-specific password
     - sender : str
        Sender email address
     - recipients : list[str]
        List of recipient email addresses
     - subject : str
        Email subject line
     - body : str
        Plain-text email body
     - image_path : str or Path
        Path to image file to attach (PNG, JPG, etc.)
    """

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Attachment not found: {image_path}")

    # Infer MIME type
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError("Could not determine MIME type for image")

    maintype, subtype = mime_type.split("/")

    # Construct email
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.set_content(body)

    # Attach image
    with open(image_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype=maintype,
            subtype=subtype,
            filename=image_path.name,
        )

    # Send email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(username, password)
        server.send_message(msg)

def send_email_html_inline_image(
    *,
    smtp_server: str,
    smtp_port: int,
    username: str,
    password: str,
    sender: str,
    recipients: list[str],
    subject: str,
    text_body: str,
    html_body: str,
    inline_image_path: Path | str | None = None,
    attachments: Optional[Iterable[Path | str]] = None,
):
    """
    Send an email with:
      - plain-text fallback
      - HTML body
      - optional inline image (CID)
      - optional multiple file attachments
    """

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject

    # Plain text fallback
    msg.set_content(text_body)

    # HTML version
    msg.add_alternative(html_body, subtype="html")

    # Inline image
    if inline_image_path is not None:
        inline_image_path = Path(inline_image_path)
        if not inline_image_path.exists():
            raise FileNotFoundError(f"Inline image not found: {inline_image_path}")

        mime_type, _ = mimetypes.guess_type(inline_image_path)
        if mime_type is None:
            raise ValueError("Could not infer MIME type for inline image")

        maintype, subtype = mime_type.split("/")

        with open(inline_image_path, "rb") as f:
            msg.get_payload()[-1].add_related(
                f.read(),
                maintype=maintype,
                subtype=subtype,
                cid="image1",   # referenced in HTML as cid:image1
            )

    # File attachments
    if attachments:
        for attachment in attachments:
            attachment = Path(attachment)
            if not attachment.exists():
                raise FileNotFoundError(f"Attachment not found: {attachment}")

            mime_type, _ = mimetypes.guess_type(attachment)
            if mime_type is None:
                raise ValueError(f"Could not infer MIME type for {attachment}")

            maintype, subtype = mime_type.split("/")

            with open(attachment, "rb") as f:
                msg.add_attachment(
                    f.read(),
                    maintype=maintype,
                    subtype=subtype,
                    filename=attachment.name,
                )

    # Send
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(username, password)
        server.send_message(msg)

@log_exceptions_with_retry(
    max_retries=5,
    prefix_fn=debug_print,
    retry_delay=1.0,   # optional
)
def build_roi_email_with_action_images(
    roi_data: dict,
    image_dir: Path | str | None = None,
    image_map: dict[str, Path | str] | None = None,
):
    """
    Build plain-text and HTML email bodies for ROI notifications, including
    optional per-action inline images.

    Args:
        roi_data (dict):
            Mapping of action ticker → ROI metadata dictionary.
            Required fields per action:
              - ACTION (str): action identifier including currency (e.g. "AAL.L_GBP→GBP")
              - SET ROI TARGET (float): target ROI (e.g. 0.1 for 10%)
              - PURCHASE PRICE (float): buy price
              - EXIT ACTION PRICE (float): sell price at ROI target
              - DATE TO ACHIEVE TARGET (Timedelta): elapsed time to reach ROI

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

    now = datetime.today().strftime("%d-%m-%Y %H:%M")
    text_lines = [f"Data extracted on {now}\n"]
    table_rows = []
    image_sections = []
    inline_images = {}

    image_dir = Path(image_dir) if image_dir else None

    for action, data in roi_data.items():
        currency = data["ACTION"].split("→")[1]
        cid = f"{action}_img"

        # --- TEXT BODY ---
        text_lines.extend([
            f"Action: {action}",
            f"  Target ROI: {data['SET ROI TARGET'] * 100:.1f}%",
            f"  Purchase: {data['PURCHASE PRICE']:.2f} {currency}",
            f"  Exit: {data['EXIT ACTION PRICE']:.2f} {currency}",
            f"  Time to target: {data['DATE TO ACHIEVE TARGET']}",
            "",
        ])

        # --- TABLE ---
        table_rows.append(f"""
        <tr>
            <td><b>{action}</b></td>
            <td>{data['SET ROI TARGET'] * 100:.1f}%</td>
            <td>{data['PURCHASE PRICE']:.2f} {currency}</td>
            <td>{data['PURCHASE DATE'].strftime('%d-%m-%Y %H:%M')}</td>
            <td>{data['EXIT ACTION PRICE']:.2f} {currency}</td>
            <td>{data['DATE TARGET MET'].strftime('%d-%m-%Y %H:%M')}</td>
            <td>{data['DATE TO ACHIEVE TARGET']}</td>
        </tr>
        """)

        # --- IMAGE RESOLUTION ---
        img_path = None
        if image_map and action in image_map:
            img_path = image_map[action]
        elif image_dir:
            candidate = image_dir / f"{data['ACTION']}_ROI.png"
            # candidate = os.path.join(image_dir,f"{data['ACTION']}_ROI.png")
            if candidate.exists():
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
        <h2>ROI targets reached</h2>
        <p>Data extracted on <b>{now}</b></p>

        <table border="1" cellpadding="6" cellspacing="0">
            <tr style="background:#f0f0f0;">
                <th>Action</th>
                <th>Target ROI</th>
                <th>Buy Price</th>
                <th>Buy Date</th>
                <th>Sell Price</th>
                <th>Sell Date</th>
                <th>Time to Target</th>
            </tr>
            {''.join(table_rows)}
        </table>

        <hr>
        {''.join(image_sections)}
    </body>
    </html>
    """

    return "\n".join(text_lines), html_body, inline_images

@log_exceptions_with_retry(
    max_retries=5,
    prefix_fn=debug_print,
    retry_delay=1.0,   # optional
)
def send_email_html_multi_inline_images(
    *,
    smtp_server: str,
    smtp_port: int,
    username: str,
    password: str,
    sender: str,
    recipients: list[str],
    subject: str,
    text_body: str,
    html_body: str,
    inline_images: dict[str, Path | str],  # {cid: path}
    attachments: Optional[Iterable[Path | str]] = None,
):
    """
    Send an email with:
      - plain-text fallback
      - HTML body
      - multiple inline images via CID
      - optional file attachments
    """

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject

    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")

    html_part = msg.get_payload()[-1]

    # Inline images
    for cid, path in inline_images.items():
        path = Path(path)
        if not path.exists():
            continue  # do not break email if one image is missing

        mime_type, _ = mimetypes.guess_type(path)
        if mime_type is None:
            continue

        maintype, subtype = mime_type.split("/")

        with open(path, "rb") as f:
            html_part.add_related(
                f.read(),
                maintype=maintype,
                subtype=subtype,
                cid=cid,
            )

    # Attachments (optional)
    if attachments:
        for attachment in attachments:
            attachment = Path(attachment)
            if not attachment.exists():
                continue

            mime_type, _ = mimetypes.guess_type(attachment)
            if mime_type is None:
                continue

            maintype, subtype = mime_type.split("/")

            with open(attachment, "rb") as f:
                msg.add_attachment(
                    f.read(),
                    maintype=maintype,
                    subtype=subtype,
                    filename=attachment.name,
                )

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(username, password)
        server.send_message(msg)


if __name__ == "__main__":
    load_dotenv()
    email_sender = os.getenv("EMAIL_SENDER")
    email_user = os.getenv("EMAIL_USER")
    email_sender_psw = os.getenv("EMAIL_SENDER_PSW")
    file_example = {   
    'AAL': {   'ACTION': 'AAL.L_GBP→GBP',
            'DATE TARGET MET': pd.Timestamp('2025-09-09 00:00:00+0100', tz='Europe/London'),
            'DATE TO ACHIEVE TARGET': pd.Timedelta('100 days 00:00:00'),
            'EXIT ACTION PRICE': 2490.0,
            'PURCHASE DATE': pd.Timestamp('2025-06-01 00:00:00+0100', tz='Europe/London'),
            'PURCHASE PRICE': 2220.0,
            'SET ROI TARGET': 0.1},
    'BRBY': {   'ACTION': 'BRBY.L_GBP→GBP',
                'DATE TARGET MET': pd.Timestamp('2025-06-27 00:00:00+0100', tz='Europe/London'),
                'DATE TO ACHIEVE TARGET': pd.Timedelta('26 days 00:00:00'),
                'EXIT ACTION PRICE': 1150.0,
                'PURCHASE DATE': pd.Timestamp('2025-06-01 00:00:00+0100', tz='Europe/London'),
                'PURCHASE PRICE': 1045.0,
                'SET ROI TARGET': 0.1},
    'CNA': {   'ACTION': 'CNA.L_GBP→GBP',
               'DATE TARGET MET': pd.Timestamp('2025-10-14 00:00:00+0100', tz='Europe/London'),
               'DATE TO ACHIEVE TARGET': pd.Timedelta('135 days 00:00:00'),
               'EXIT ACTION PRICE': 173.0,
               'PURCHASE DATE': pd.Timestamp('2025-06-01 00:00:00+0100', tz='Europe/London'),
               'PURCHASE PRICE': 157.14999389648438,
               'SET ROI TARGET': 0.1},
    'ENT': {   'ACTION': 'ENT.L_GBP→GBP',
               'DATE TARGET MET': pd.Timestamp('2025-06-16 00:00:00+0100', tz='Europe/London'),
               'DATE TO ACHIEVE TARGET': pd.Timedelta('15 days 00:00:00'),
               'EXIT ACTION PRICE': 866.0,
               'PURCHASE DATE': pd.Timestamp('2025-06-01 00:00:00+0100', tz='Europe/London'),
               'PURCHASE PRICE': 749.4000244140625,
               'SET ROI TARGET': 0.1},
    'GLEN': {   'ACTION': 'GLEN.L_GBP→GBP',
                'DATE TARGET MET': pd.Timestamp('2025-07-23 00:00:00+0100', tz='Europe/London'),
                'DATE TO ACHIEVE TARGET': pd.Timedelta('52 days 00:00:00'),
                'EXIT ACTION PRICE': 326.45001220703125,
                'PURCHASE DATE': pd.Timestamp('2025-06-01 00:00:00+0100', tz='Europe/London'),
                'PURCHASE PRICE': 284.79998779296875,
                'SET ROI TARGET': 0.1},
    'MNG': {   'ACTION': 'MNG.L_GBP→GBP',
               'DATE TARGET MET': pd.Timestamp('2025-08-11 00:00:00+0100', tz='Europe/London'),
               'DATE TO ACHIEVE TARGET': pd.Timedelta('71 days 00:00:00'),
               'EXIT ACTION PRICE': 263.6000061035156,
               'PURCHASE DATE': pd.Timestamp('2025-06-01 00:00:00+0100', tz='Europe/London'),
               'PURCHASE PRICE': 239.39999389648438,
               'SET ROI TARGET': 0.1},
    'PHNX': {   'ACTION': 'PHNX.L_GBP→GBP',
                'DATE TARGET MET': pd.Timestamp('2025-12-17 00:00:00+0000', tz='Europe/London'),
                'DATE TO ACHIEVE TARGET': pd.Timedelta('199 days 01:00:00'),
                'EXIT ACTION PRICE': 719.0,
                'PURCHASE DATE': pd.Timestamp('2025-06-01 00:00:00+0100', tz='Europe/London'),
                'PURCHASE PRICE': 644.5,
                'SET ROI TARGET': 0.1},
    'VOD': {   'ACTION': 'VOD.L_GBP→GBP',
               'DATE TARGET MET': pd.Timestamp('2025-07-24 00:00:00+0100', tz='Europe/London'),
               'DATE TO ACHIEVE TARGET': pd.Timedelta('53 days 00:00:00'),
               'EXIT ACTION PRICE': 86.0199966430664,
               'PURCHASE DATE': pd.Timestamp('2025-06-01 00:00:00+0100', tz='Europe/London'),
               'PURCHASE PRICE': 76.77999877929688,
               'SET ROI TARGET': 0.1}
               }
    
    body = f"Data extracted on {datetime.today().strftime("%d-%m-%Y %H:%M")}\n"
    for ac, data in file_example.items():
        try:
            currency = data['ACTION'].split("→")[1]
            body += f"\n    Action: {data['ACTION']} - target ROI:{data['SET ROI TARGET']*100}%"
            body += f"\n        Purchase Date: {data['PURCHASE DATE'].strftime("%d-%m-%Y %H:%M")}"
            body += f"\n        Purchase Price: {round(data['PURCHASE PRICE'], 2)} {currency}"
            body += f"\n        Suggested Sell Date: {data['DATE TARGET MET'].strftime("%d-%m-%Y %H:%M")}"
            body += f"\n        Suggested Sell Price: {round(data['EXIT ACTION PRICE'], 2)} {currency}"
            body += f"\n        Suggested Sell Date: {data['DATE TARGET MET'].strftime("%d-%m-%Y %H:%M")}"           
            body += f"\n        Days to achieve set ROI: {data['DATE TO ACHIEVE TARGET']}\n\n"
            

        except Exception as e:
            print(f"ERROR constructing body for email for action {ac} {type(e).__name__}: {e}")

    old_email_structure = False

    if old_email_structure == True:
        try:
            send_email_with_image(
                smtp_server="smtp.gmail.com",
                smtp_port=587,
                username=email_sender,
                password=email_sender_psw,
                sender=email_sender,
                recipients=["ingcarldan@gmail.com"],
                subject=f"ROI target reached on {datetime.today().strftime("%d-%m-%Y %H:%M")}",
                body=body,
                image_path="output.png",
            )
        except Exception as e:
            print(f"ERROR {type(e).__name__}: {e}")
    elif old_email_structure == False:
        pics_dir = os.path.join(os.getcwd(),"output")
        text_body, html_body, inline_images = build_roi_email_with_action_images(
            roi_data=file_example,
            image_dir=pics_dir,  # contains AAL.png, VOD.png, ...
        )
        
        send_email_html_multi_inline_images(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            username=email_sender,
            password=email_sender_psw,
            sender=email_sender,
            recipients=["ingcarldan@gmail.com"],
            subject=f"ROI targets reached – {datetime.today():%d-%m-%Y %H:%M}",
            text_body=text_body,
            html_body=html_body,
            inline_images=inline_images,
        )


"""
NOTE : 
    -   Add HTML + inline image
    -   Integrate this directly into your ROI trigger loop
    -   Provide a config-driven version (YAML / env vars)
    -   Add multiple attachments
"""