import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
import mimetypes
from dotenv import load_dotenv

load_dotenv()
email_sender = os.getenv("EMAIL_SENDER")
email_sender_psw = os.getenv("EMAIL_SENDER_PSW")


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

if __name__ == "__main__":
    load_dotenv()
    email_sender = os.getenv("EMAIL_SENDER")
    email_user = os.getenv("EMAIL_USER")
    email_sender_psw = os.getenv("EMAIL_SENDER_PSW")

    try:
        send_email_with_image(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            username=email_sender,
            password=email_sender_psw,
            sender=email_sender,
            recipients=["ingcarldan@gmail.com"],
            subject="ROI target reached",
            body="The attached chart shows the BUY and ROI exit points.",
            image_path="output.png",
        )
    except Exception as e:
        print(f"ERROR {type(e).__name__}: {e}")


"""
NOTE : 
    -   Add HTML + inline image
    -   Integrate this directly into your ROI trigger loop
    -   Provide a config-driven version (YAML / env vars)
    -   Add multiple attachments
"""