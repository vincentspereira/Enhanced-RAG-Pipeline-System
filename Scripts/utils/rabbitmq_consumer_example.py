import pika
import os
import logging
import time
import json
from typing import Union, Dict

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import config loader
try:
    from Scripts.utils.config_loader import get_config_value
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Assumes this is in Scripts/utils
    try:
        from utils.config_loader import get_config_value
    except ImportError as e:
        logger.error(f"Critical: Failed to import get_config_value for RabbitMQ Consumer. Error: {e}")
        def get_config_value(env_var_name, yaml_path=None, default=None): # Basic fallback
            return os.getenv(env_var_name, default)

# RabbitMQ Configuration (should match producer's)
RABBITMQ_HOST = get_config_value("RABBITMQ_HOST", yaml_path="message_queue.rabbitmq.host", default="localhost")
RABBITMQ_PORT = int(get_config_value("RABBITMQ_PORT", yaml_path="message_queue.rabbitmq.port", default=5672))
RABBITMQ_USER = get_config_value("RABBITMQ_USER", yaml_path="message_queue.rabbitmq.user", default="user")
RABBITMQ_PASS = get_config_value("RABBITMQ_PASSWORD", yaml_path="message_queue.rabbitmq.password", default="password")

EXAMPLE_QUEUE_NAME = 'example_task_queue'

def message_processor(body_bytes: bytes) -> Union[Dict, str]:
    """Processes the raw message body."""
    try:
        # Try to decode as JSON
        message_str = body_bytes.decode('utf-8')
        data = json.loads(message_str)
        return data
    except json.JSONDecodeError:
        # If not JSON, assume it's a plain string
        return message_str
    except UnicodeDecodeError:
        logger.error("Failed to decode message body as UTF-8.")
        return f"Undecodable message (binary or wrong encoding): {body_bytes[:50]}..."


def callback(ch, method, properties, body):
    """
    This function is called when a message is received.
    ch: channel
    method: delivery_info (includes delivery_tag)
    properties: message properties (e.g., content_type, delivery_mode)
    body: message content (bytes)
    """
    logger.info(f" [x] Received message (delivery_tag: {method.delivery_tag})")
    logger.info(f"     Properties: {properties}")

    processed_body = message_processor(body)
    if isinstance(processed_body, dict):
        logger.info(f"     Body (JSON): {json.dumps(processed_body, indent=2)}")
    else:
        logger.info(f"     Body (string): {processed_body}")

    # Simulate some work being done
    # time.sleep(body.count(b'.')) # Example from Pika docs: sleep based on dots in message
    time.sleep(1) # Simulate 1 second of work

    # Acknowledge the message (tells RabbitMQ that the message has been processed)
    # This is crucial for message durability and preventing loss if consumer crashes.
    ch.basic_ack(delivery_tag=method.delivery_tag)
    logger.info(f" [x] Done processing message (delivery_tag: {method.delivery_tag}). Acknowledged.")


def start_consuming():
    connection = None
    try:
        credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
        parameters = pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            credentials=credentials
        )
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()

        # Declare the queue (ensure it exists and is durable, same as producer)
        channel.queue_declare(queue=EXAMPLE_QUEUE_NAME, durable=True)
        logger.info(f"[*] Waiting for messages in queue '{EXAMPLE_QUEUE_NAME}'. To exit press CTRL+C")

        # Fair dispatch: Don't give more than one message to a worker at a time.
        # If a worker is busy, messages go to other available workers.
        channel.basic_qos(prefetch_count=1)

        # Set up subscription on the queue
        channel.basic_consume(
            queue=EXAMPLE_QUEUE_NAME,
            on_message_callback=callback
            # auto_ack=False # Default, manual acknowledgement is done in `callback`
        )

        # Start consuming messages. This is a blocking call.
        channel.start_consuming()

    except pika.exceptions.AMQPConnectionError as e:
        logger.error(f"Failed to connect to RabbitMQ at {RABBITMQ_HOST}:{RABBITMQ_PORT}. Error: {e}")
        logger.info("Retrying in 5 seconds...")
        time.sleep(5)
        start_consuming() # Simple retry logic for example
    except KeyboardInterrupt:
        logger.info("Consumer stopped by user (CTRL+C).")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        if connection and connection.is_open:
            try:
                connection.close()
                logger.info("RabbitMQ connection closed.")
            except Exception as e_close:
                logger.error(f"Error closing RabbitMQ connection: {e_close}")

if __name__ == '__main__':
    start_consuming()

    # To run this consumer:
    # 1. Ensure RabbitMQ server is running and accessible (same as producer).
    # 2. Set environment variables if defaults are not suitable (same as producer).
    # 3. Run the script: python Scripts/utils/rabbitmq_consumer_example.py
    #
    # This script will connect to RabbitMQ, subscribe to 'example_task_queue',
    # and wait for messages. When a message arrives, the `callback` function
    # will process it, simulate work, and then acknowledge it.
    #
    # Run the producer script (rabbitmq_producer_example.py) in another terminal
    # to send messages that this consumer will pick up.
