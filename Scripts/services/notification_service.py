import os
import smtplib
from email.mime.text import MIMEText
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, EmailStr
import logging
from typing import List

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = FastAPI(title="Notification Service")

# --- Configuration (loaded from environment variables for simplicity) ---
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")

class EmailSchema(BaseModel):
    to_emails: List[EmailStr]
    subject: str
    body: str

def send_email_background(to_emails: List[EmailStr], subject: str, body: str):
    """
    Sends an email in the background.
    Note: Error handling for actual email sending should be robust in production.
    """
    if not all([SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SENDER_EMAIL]):
        logger.error("SMTP server settings are not fully configured. Cannot send email.")
        # In a real scenario, you might raise an exception or handle this more gracefully.
        return

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = ", ".join(to_emails)

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Use TLS
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SENDER_EMAIL, to_emails, msg.as_string())
        logger.info(f"Email successfully sent to: {', '.join(to_emails)}")
    except smtplib.SMTPAuthenticationError:
        logger.error("SMTP Authentication Error. Please check username/password.")
    except smtplib.SMTPServerDisconnected:
        logger.error("SMTP server disconnected unexpectedly.")
    except smtplib.SMTPConnectError:
        logger.error(f"Could not connect to SMTP server: {SMTP_SERVER}:{SMTP_PORT}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")

@app.post("/send_email")
async def trigger_send_email(email: EmailSchema, background_tasks: BackgroundTasks):
    if not all([SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SENDER_EMAIL]):
        raise HTTPException(status_code=500, detail="SMTP settings not configured on the server.")

    background_tasks.add_task(send_email_background, email.to_emails, email.subject, email.body)
    return {"message": "Email sending process initiated."}

@app.get("/health", tags=["Health"])
async def health_check():
    # Basic health check, can be expanded (e.g., check SMTP config presence)
    return {"status": "healthy", "service": "Notification Service"}

if __name__ == "__main__":
    import uvicorn
    # Example: Set environment variables for local testing
    # os.environ["SMTP_SERVER"] = "smtp.example.com"
    # os.environ["SMTP_PORT"] = "587"
    # os.environ["SMTP_USERNAME"] = "user@example.com"
    # os.environ["SMTP_PASSWORD"] = "password"
    # os.environ["SENDER_EMAIL"] = "noreply@example.com"

    if not all([SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SENDER_EMAIL]):
        logger.warning("One or more SMTP environment variables are not set. Email sending will fail.")
        logger.warning("Please set: SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SENDER_EMAIL")

    SERVICE_PORT = int(os.getenv("NOTIFICATION_SERVICE_PORT", 8003))
    SERVICE_HOST = os.getenv("NOTIFICATION_SERVICE_HOST", "0.0.0.0")

    logger.info(f"Starting Notification Service on {SERVICE_HOST}:{SERVICE_PORT}")
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT)
