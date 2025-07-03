import pika
import logging
import json
import time
import os
from typing import Optional

# Configure logger
# It's better for a script/application to configure its own logging if run standalone.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# RabbitMQ connection parameters (can be loaded from config or environment variables)
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", 5672))
RABBITMQ_USER = os.getenv("RABBITMQ_USER") # Optional
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD") # Optional
RABBITMQ_VHOST = os.getenv("RABBITMQ_VHOST", "/")
QUEUE_NAME = 'document_processing_events'

class DocumentEventConsumer:
    def __init__(self, host: str, port: int, username: Optional[str], password: Optional[str], virtual_host: str, queue_name: str):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.virtual_host = virtual_host
        self.queue_name = queue_name

        self.connection: Optional[pika.BlockingConnection] = None
        self.channel: Optional[pika.adapters.blocking_connection.BlockingChannel] = None

    def _connect(self):
        logger.info(f"Attempting to connect to RabbitMQ at {self.host}:{self.port}...")
        credentials = None
        if self.username and self.password:
            credentials = pika.PlainCredentials(self.username, self.password)

        parameters = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            virtual_host=self.virtual_host,
            credentials=credentials,
            heartbeat=600,
            blocked_connection_timeout=300
        )
        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()

        # Declare the exchange (topic exchange)
        self.exchange_name = 'document_events_exchange' # Should match producer
        self.channel.exchange_declare(exchange=self.exchange_name, exchange_type='topic', durable=True)
        logger.info(f"Topic exchange '{self.exchange_name}' ensured.")

        # Declare the queue this consumer will use.
        # For DLQ: arguments={'x-dead-letter-exchange': 'my_dlx', 'x-dead-letter-routing-key': 'dlq_key'}
        # For Priority Queue: arguments={'x-max-priority': 10} (Value 1-255, typically 1-10)
        # These arguments are set during queue declaration on the RabbitMQ server, often by an admin or deployment script.
        # Client can declare queue with these args if it has permissions and queue doesn't exist with different args.
        queue_arguments = {}
        # Example if setting up DLQ (ensure 'my_dlx' exchange exists)
        # queue_arguments['x-dead-letter-exchange'] = 'my_dlx'
        # Example for priority (ensure producer sends messages with priority property)
        # queue_arguments['x-max-priority'] = 10

        self.channel.queue_declare(queue=self.queue_name, durable=True, arguments=queue_arguments)
        logger.info(f"Queue '{self.queue_name}' declared (durable=True). Arguments: {queue_arguments}")

        # Bind the queue to the exchange with a routing key pattern
        # Example: "doc.processed.*" to receive all processed document events regardless of file type.
        # Or "doc.processed.pdf" to only receive events for PDF files.
        binding_key = "doc.processed.*" # Listen to all doc.processed events
        self.channel.queue_bind(exchange=self.exchange_name, queue=self.queue_name, routing_key=binding_key)
        logger.info(f"Queue '{self.queue_name}' bound to exchange '{self.exchange_name}' with binding key '{binding_key}'.")

        logger.info(f"RabbitMQ Consumer connected and setup complete for queue '{self.queue_name}'.")

    def _message_callback(self, ch: pika.adapters.blocking_connection.BlockingChannel,
                          method: pika.spec.Basic.Deliver,
                          properties: pika.spec.BasicProperties,
                          body: bytes):
        """Callback function to process messages from the queue."""
        try:
            message_str = body.decode('utf-8')
            # Log received message priority if present
            priority = properties.priority if properties and hasattr(properties, 'priority') else 'N/A'
            logger.info(f"Received message (Priority: {priority}, RoutingKey: {method.routing_key}): {message_str[:200]}...")

            message_data = json.loads(message_str)

            # --- Process the message based on its content ---
            event_type = message_data.get("event_type")
            if event_type == "document_processed":
                doc_id = message_data.get("parent_document_id")
                filename = message_data.get("filename", "N/A")
                chunks = message_data.get("chunks_indexed", 0)
                status = message_data.get("status", "N/A")
                logger.info(f"Processing 'document_processed' event: ID={doc_id}, File={filename}, Chunks={chunks}, Status={status}")
                # Add actual processing logic here
            else:
                logger.warning(f"Received unknown event_type: '{event_type}' from routing key '{method.routing_key}'")

            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.debug(f"Message acknowledged: {method.delivery_tag}")

        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON message: {body[:200]}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    def start_consuming(self):
        """Start consuming messages from the queue."""
        while True: # Keep trying to connect and consume
            try:
                self._connect()
                if not self.channel:
                    logger.error("Cannot start consuming: RabbitMQ channel is not available.")
                    time.sleep(10) # Wait before retrying connection
                    continue

                # Set Quality of Service: process one message at a time.
                # This is important if processing is resource-intensive or long.
                self.channel.basic_qos(prefetch_count=1)

                # Start consuming
                self.channel.basic_consume(
                    queue=self.queue_name,
                    on_message_callback=self._message_callback
                    # auto_ack=False # Explicit acknowledgement is handled in _message_callback
                )

                logger.info(f"Waiting for messages on queue '{self.queue_name}'. To exit press CTRL+C")
                self.channel.start_consuming() # This is a blocking call

            except pika.exceptions.AMQPConnectionError as conn_err:
                logger.error(f"RabbitMQ connection error: {conn_err}. Retrying in 10 seconds...")
                self._close_connection_gracefully()
                time.sleep(10)
            except pika.exceptions.StreamLostError as stream_err:
                logger.error(f"RabbitMQ stream lost: {stream_err}. Retrying in 10 seconds...")
                self._close_connection_gracefully()
                time.sleep(10)
            except KeyboardInterrupt:
                logger.info("Consumer stopped by user (CTRL+C).")
                break
            except Exception as e:
                logger.error(f"An unexpected error occurred in consumer: {e}", exc_info=True)
                self._close_connection_gracefully()
                logger.info("Attempting to restart consumer in 30 seconds...")
                time.sleep(30) # Longer delay for unexpected errors
            finally:
                self._close_connection_gracefully()

        logger.info("Consumer shutdown complete.")

    def _close_connection_gracefully(self):
        """Safely close the connection and channel."""
        try:
            if self.channel and self.channel.is_open:
                self.channel.close()
                logger.info("RabbitMQ channel closed.")
        except Exception as e:
            logger.error(f"Error closing RabbitMQ channel: {e}")
        try:
            if self.connection and self.connection.is_open:
                self.connection.close()
                logger.info("RabbitMQ connection closed.")
        except Exception as e:
            logger.error(f"Error closing RabbitMQ connection: {e}")
        finally:
            self.channel = None
            self.connection = None

if __name__ == "__main__":
    logger.info("Starting Document Event Consumer...")
    # Ensure RabbitMQ is running and accessible.
    # If RabbitMQ requires authentication, set RABBITMQ_USER and RABBITMQ_PASSWORD environment variables.
    # e.g., export RABBITMQ_HOST=myrabbit.example.com
    # e.g., export RABBITMQ_USER=user
    # e.g., export RABBITMQ_PASSWORD=password

    consumer = DocumentEventConsumer(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        username=RABBITMQ_USER,
        password=RABBITMQ_PASSWORD,
        virtual_host=RABBITMQ_VHOST,
        queue_name=QUEUE_NAME
    )
    consumer.start_consuming()
    logger.info("Document Event Consumer finished.")
