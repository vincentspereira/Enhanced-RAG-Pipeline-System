import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from typing import List

# Assuming notification_service.py is in Scripts/services/
# Adjust the import path if necessary based on your project structure and how pytest discovers modules.
# If running pytest from the root, 'Scripts.services.notification_service' should work if 'Scripts' is a package (has __init__.py)
# or if PYTHONPATH is set up. For simplicity, direct import path relative to a common root might be assumed by pytest.
from Scripts.services.notification_service import app, send_email_background, EmailSchema

client = TestClient(app)

# Target for patching os.getenv, used by the notification_service module at import time for SMTP config
OS_GETENV_TARGET = "Scripts.services.notification_service.os.getenv"
SMTPLIB_SMTP_TARGET = "Scripts.services.notification_service.smtplib.SMTP"


@pytest.fixture(autouse=True) # Apply to all tests in this module
def mock_smtp_env_vars(mocker):
    """Mocks environment variables for SMTP configuration for all tests."""
    env_map = {
        "SMTP_SERVER": "smtp.mockserver.com",
        "SMTP_PORT": "587", # os.getenv returns string, service converts to int
        "SMTP_USERNAME": "mock_user",
        "SMTP_PASSWORD": "mock_password",
        "SENDER_EMAIL": "sender@mock.com",
    }
    mocker.patch(OS_GETENV_TARGET, side_effect=lambda key, default=None: env_map.get(key, default))

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "Notification Service"}

@patch(SMTPLIB_SMTP_TARGET) # Mock the smtplib.SMTP class
def test_send_email_endpoint_success(mock_smtp_class, mock_smtp_env_vars):
    """Test the /send_email endpoint for successful email initiation."""
    mock_smtp_instance = MagicMock()
    mock_smtp_class.return_value.__enter__.return_value = mock_smtp_instance # Mock the context manager

    email_data = {
        "to_emails": ["recipient1@example.com", "recipient2@example.com"],
        "subject": "Test Subject",
        "body": "Test email body."
    }
    response = client.post("/send_email", json=email_data)

    assert response.status_code == 200
    assert response.json() == {"message": "Email sending process initiated."}

    # Background task makes direct assertions on mock_smtp_instance tricky here
    # without more complex background task testing.
    # We'd typically test send_email_background directly for SMTP interactions.

def test_send_email_endpoint_smtp_not_configured():
    """Test /send_email when SMTP settings are missing (mocking os.getenv to return None)."""
    with patch(OS_GETENV_TARGET, return_value=None): # Override the autouse fixture for this test
        # Re-import or re-initialize app if SMTP vars are read at import/startup globally in service
        # For this test, we assume the endpoint checks on each call or global vars are affected by patch
        email_data = {
            "to_emails": ["recipient@example.com"],
            "subject": "Test Subject",
            "body": "Test body."
        }
        response = client.post("/send_email", json=email_data)
        assert response.status_code == 500
        assert "SMTP settings not configured" in response.json()["detail"]


@patch(SMTPLIB_SMTP_TARGET)
def test_send_email_background_function_success(mock_smtp_class, mock_smtp_env_vars):
    """Test the send_email_background function directly for successful sending."""
    mock_smtp_instance = MagicMock()
    mock_smtp_class.return_value.__enter__.return_value = mock_smtp_instance

    to_emails: List[str] = ["test@example.com"]
    subject = "Background Test"
    body = "Background Body"

    send_email_background(to_emails, subject, body)

    mock_smtp_class.assert_called_once_with("smtp.mockserver.com", 587)
    mock_smtp_instance.starttls.assert_called_once()
    mock_smtp_instance.login.assert_called_once_with("mock_user", "mock_password")
    mock_smtp_instance.sendmail.assert_called_once()

    # Check email content passed to sendmail
    args, _ = mock_smtp_instance.sendmail.call_args
    assert args[0] == "sender@mock.com"
    assert args[1] == to_emails
    # args[2] is the message string. Check some key parts.
    assert f"Subject: {subject}" in args[2]
    assert f"From: sender@mock.com" in args[2]
    assert f"To: {', '.join(to_emails)}" in args[2]
    assert body in args[2]


@patch(SMTPLIB_SMTP_TARGET)
def test_send_email_background_smtp_auth_error(mock_smtp_class, mock_smtp_env_vars, caplog):
    """Test send_email_background with SMTPAuthenticationError."""
    mock_smtp_instance = MagicMock()
    mock_smtp_class.return_value.__enter__.return_value = mock_smtp_instance
    mock_smtp_instance.login.side_effect = smtplib.SMTPAuthenticationError(535, "Auth failed")

    with caplog.at_level("ERROR"):
        send_email_background(["test@example.com"], "Auth Error Test", "Body")

    assert "SMTP Authentication Error" in caplog.text
    mock_smtp_instance.sendmail.assert_not_called()


@patch(OS_GETENV_TARGET, return_value=None) # Mock os.getenv to return None for all SMTP vars
def test_send_email_background_smtp_not_configured_direct_call(mock_getenv_none, caplog):
    """Test send_email_background directly when SMTP is not configured."""
    # This test relies on send_email_background re-checking its globals or os.getenv itself.
    # The global SMTP_SERVER etc. in notification_service might be set at import time.
    # To make this test robust, notification_service.py might need to fetch env vars inside send_email_background
    # or be reloaded with the patch. For now, assuming it checks.

    # If SMTP settings are module-level globals set on import, this test might not reflect
    # the function's behavior if it doesn't re-evaluate those globals.
    # A better way would be to pass config into the function or class.

    # Forcing the global vars used by send_email_background to None for this test scope
    with patch("Scripts.services.notification_service.SMTP_SERVER", None), \
         patch("Scripts.services.notification_service.SMTP_PORT", None), \
         patch("Scripts.services.notification_service.SMTP_USERNAME", None), \
         patch("Scripts.services.notification_service.SMTP_PASSWORD", None), \
         patch("Scripts.services.notification_service.SENDER_EMAIL", None):

        with caplog.at_level("ERROR"):
            send_email_background(["test@example.com"], "No Config Test", "Body")

        assert "SMTP server settings are not fully configured" in caplog.text

# Add more tests for other SMTP exceptions (SMTPServerDisconnected, SMTPConnectError) if desired.
# Example for SMTPConnectError:
@patch(SMTPLIB_SMTP_TARGET)
def test_send_email_background_smtp_connect_error(mock_smtp_class, mock_smtp_env_vars, caplog):
    """Test send_email_background with SMTPConnectError."""
    mock_smtp_class.side_effect = smtplib.SMTPConnectError(None, "Connection refused")

    with caplog.at_level("ERROR"):
        send_email_background(["test@example.com"], "Connect Error Test", "Body")

    assert "Could not connect to SMTP server" in caplog.text
    # login and sendmail should not be called
    # Need to access the instance created by the context manager to check this.
    # This test setup makes it tricky. A direct mock_smtp_instance would be better.
    # For now, relying on the log message.
