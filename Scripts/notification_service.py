import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class Notifier(ABC):
    """Abstract base class for notification services."""

    @abstractmethod
    async def send_notification(
        self,
        subject: str,
        message: str,
        recipient: Optional[str] = None, # e.g., email address, Slack channel/user ID
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Sends a notification.

        Args:
            subject (str): The subject/title of the notification.
            message (str): The main content of the notification.
            recipient (Optional[str]): The intended recipient or channel.
                                       Interpretation depends on the notifier type.
            metadata (Optional[Dict[str, Any]]): Additional context or data for the notification.

        Returns:
            bool: True if the notification was sent successfully, False otherwise.
        """
        pass

class LoggingNotifier(Notifier):
    """A notifier that logs notifications using the standard logging module."""

    def __init__(self, level: int = logging.INFO):
        self.level = level
        # Ensure the logger used by this notifier is configured.
        # If this module is part of a larger app, app-level logging config should cover it.
        # For standalone use, basicConfig might be needed here or in the calling code.
        self.notifier_logger = logging.getLogger("NotificationService.LoggingNotifier")
        if not self.notifier_logger.handlers: # Add a default handler if none configured
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - Notification - Subject: "%(subject)s" - Message: "%(message)s"')
            handler.setFormatter(formatter)
            self.notifier_logger.addHandler(handler)
            self.notifier_logger.setLevel(self.level)
            self.notifier_logger.propagate = False # Avoid duplicate logs if root logger also has handlers

    async def send_notification(
        self,
        subject: str,
        message: str,
        recipient: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        log_message = f"Recipient: {recipient or 'N/A'} | Subject: '{subject}' | Message: '{message}'"
        if metadata:
            log_message += f" | Metadata: {metadata}"

        # Log with specific fields for easier parsing if needed, or just as a string
        self.notifier_logger.log(self.level, log_message, extra={
            'notification_subject': subject,
            'notification_message': message,
            'notification_recipient': recipient,
            'notification_metadata': metadata
        })
        # For the purpose of this notifier, logging is considered "sending successfully"
        return True

# --- Factory Function ---
# Using 'Any' for config type to avoid direct import of ConfigManager.SystemConfig
# and potential circular dependencies if this module is imported by config-related modules.
# The calling code will be responsible for passing the correct config structure.
_notification_service_instance: Optional[Notifier] = None

def initialize_notification_service(app_config: Any):
    """
    Initializes the notification service based on application configuration.
    This should ideally be called once at application startup.
    """
    global _notification_service_instance

    if _notification_service_instance is not None:
        logger.warning("Notification service already initialized.")
        return

    # Assuming app_config has a 'notification' attribute which is a NotificationConfig instance
    notification_config = getattr(app_config, 'notification', None)
    notifier_type = "logging" # Default

    if notification_config and hasattr(notification_config, 'notifier_type'):
        notifier_type = getattr(notification_config, 'notifier_type', "logging").lower()
        # Add other config params for specific notifiers here
        # e.g., email_host, slack_webhook_url from notification_config

    logger.info(f"Initializing notification service with type: {notifier_type}")

    if notifier_type == "logging":
        log_level_str = getattr(notification_config, 'log_level', "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        _notification_service_instance = LoggingNotifier(level=log_level)
    # elif notifier_type == "email":
    #     _notification_service_instance = EmailNotifier(host=..., port=..., ...)
    # elif notifier_type == "slack":
    #     _notification_service_instance = SlackNotifier(webhook_url=...)
    else:
        logger.warning(f"Unknown notifier type '{notifier_type}'. Defaulting to LoggingNotifier.")
        _notification_service_instance = LoggingNotifier()

    logger.info(f"Notification service initialized with {_notification_service_instance.__class__.__name__}")


def get_notification_service() -> Optional[Notifier]:
    """
    Returns the initialized notification service instance.

    Raises:
        RuntimeError: If the service has not been initialized.
    """
    if _notification_service_instance is None:
        # Option 1: Raise an error
        # raise RuntimeError("Notification service has not been initialized. Call initialize_notification_service() first.")
        # Option 2: Initialize with defaults if not done (might hide issues but can be convenient)
        logger.warning("Notification service accessed before explicit initialization. Initializing with default LoggingNotifier.")
        # This would require access to a default config or making LoggingNotifier parameterless
        # For now, let's assume it must be initialized.
        # A better approach for FastAPI apps is to initialize on startup and use Depends.
        return None # Or raise error
    return _notification_service_instance

# Example of how it might be used in a FastAPI app (conceptual)
# from fastapi import Depends, FastAPI
# from your_app_config_module import AppConfig, get_app_config # Pseudo code

# app = FastAPI()

# @app.on_event("startup")
# async def startup_event():
#     app_config = get_app_config() # Load your app config
#     initialize_notification_service(app_config)

# async def get_notifier_dependency() -> Notifier:
#     service = get_notification_service()
#     if service is None:
#         raise HTTPException(status_code=500, detail="Notification service not available.")
#     return service

# @app.post("/send_test_notification")
# async def send_test_notification(notifier: Notifier = Depends(get_notifier_dependency)):
#     await notifier.send_notification("Test Subject", "This is a test message from FastAPI.")
#     return {"status": "Notification sent (check logs)"}

if __name__ == "__main__":
    # Example of direct usage:
    async def run_example():
        # Mock app_config for standalone testing
        class MockNotificationConfig:
            notifier_type: str = "logging"
            log_level: str = "DEBUG"

        class MockAppConfig:
            notification: MockNotificationConfig = MockNotificationConfig()

        mock_config = MockAppConfig()
        initialize_notification_service(mock_config)

        notifier = get_notification_service()
        if notifier:
            await notifier.send_notification(
                "Test Event",
                "This is a test notification from standalone script.",
                recipient="user@example.com",
                metadata={"source": "test_script", "importance": "high"}
            )
            await notifier.send_notification(
                "Another Event",
                "Just a simple log message."
            )
        else:
            print("Failed to get notifier instance.")

    asyncio.run(run_example())
