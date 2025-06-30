import os
import sys
import logging
import json
import time

# Adjust path to import from parent directory if necessary
# This assumes 'Scripts' is in PYTHONPATH or this script is run from the project root.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from utils.message_queue import RabbitMQConnection # type: ignore
except ImportError:
    # This fallback might be needed if the script is run in a way that sys.path manipulation doesn't work as expected
    # or if message_queue.py is not found.
    print("Error: Could not import RabbitMQConnection from utils.message_queue. Make sure PYTHONPATH is set correctly.")
    RabbitMQConnection = None # type: ignore

# Setup basic logging for the worker
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler() # Log to stdout/stderr
    ]
)
logger = logging.getLogger("example_doc_consumer")

# --- Configuration from Environment Variables ---
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "raguser")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "ragpassword")
RABBITMQ_VHOST = os.getenv("RABBITMQ_VHOST", "/")
DOCUMENT_QUEUE_NAME = os.getenv("DOCUMENT_QUEUE_NAME", "document_processing_queue")

def process_message_callback(ch, method, properties, body):
    """
    Callback function to process a received message.
    """
    try:
        message_content = body.decode('utf-8')
        logger.info(f"Received message from queue '{method.routing_key}':")
        logger.info(f"  Delivery Tag: {method.delivery_tag}")
        logger.info(f"  Properties: {properties}")
        logger.info(f"  Body: {message_content}")

        # Example: Parse JSON if messages are expected in that format
        # try:
        #     data = json.loads(message_content)
        #     logger.info(f"  Parsed Data: {data}")
        #     # TODO: Add actual document processing logic here based on 'data'
        # except json.JSONDecodeError:
        #     logger.warning("  Message body is not valid JSON.")

        # Simulate some work
        time.sleep(1)
        logger.info("Message processed (simulated).")

        # Acknowledge the message (important!)
        ch.basic_ack(delivery_tag=method.delivery_tag)
        logger.info(f"Message {method.delivery_tag} acknowledged.")

    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        # Optionally, decide whether to nack (requeue) the message or discard it
        # For simplicity, we'll let it be unacknowledged if an error occurs,
        # RabbitMQ might redeliver it later depending on configuration.
        # ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def main():
    if not RabbitMQConnection:
        logger.error("RabbitMQConnection class not available. Exiting.")
        return

    logger.info("Starting Example Document Consumer Worker...")
    logger.info(f"Connecting to RabbitMQ at {RABBITMQ_HOST}:{RABBITMQ_PORT}{RABBITMQ_VHOST}")
    logger.info(f"Consuming from queue: {DOCUMENT_QUEUE_NAME}")

    mq_connection = RabbitMQConnection(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        username=RABBITMQ_USER,
        password=RABBITMQ_PASS,
        virtual_host=RABBITMQ_VHOST
    )

    retry_interval = 5 # seconds
    while True: # Keep trying to connect and consume
        try:
            with mq_connection: # Handles connect and close
                mq_connection.declare_queue(DOCUMENT_QUEUE_NAME, durable=True)
                mq_connection.consume_messages(
                    queue_name=DOCUMENT_QUEUE_NAME,
                    callback_function=process_message_callback,
                    auto_ack=False # Manual acknowledgement
                )
        except pika.exceptions.AMQPConnectionError as amqp_e: # type: ignore
            logger.error(f"RabbitMQ Connection Error: {amqp_e}. Retrying in {retry_interval} seconds...")
        except ConnectionError as ce: # Catch custom ConnectionError from RabbitMQConnection
             logger.error(f"RabbitMQ Connection Error (from wrapper): {ce}. Retrying in {retry_interval} seconds...")
        except KeyboardInterrupt:
            logger.info("Consumer worker shutting down (KeyboardInterrupt).")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred in consumer main loop: {e}", exc_info=True)
            logger.info(f"Retrying in {retry_interval} seconds...")

        time.sleep(retry_interval)

if __name__ == "__main__":
    # Import pika here for the AMQPConnectionError check in main()
    # This is a bit of a workaround because RabbitMQConnection might be None if import fails early.
    try:
        import pika.exceptions
    except ImportError:
        if RabbitMQConnection is not None: # Only log if we expected pika to be there via RabbitMQConnection
            logger.error("pika library not found, cannot run consumer.")
        # If RabbitMQConnection is None, the error was already printed.
        sys.exit(1)

    main()
