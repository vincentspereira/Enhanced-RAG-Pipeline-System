import pika
import logging
import json
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class RabbitMQProducer:
    def __init__(self, host: str = 'localhost', port: int = 5672,
                 username: Optional[str] = None, password: Optional[str] = None,
                 virtual_host: str = '/'):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.virtual_host = virtual_host

        self.connection: Optional[pika.BlockingConnection] = None
        self.channel: Optional[pika.adapters.blocking_connection.BlockingChannel] = None
        self._connect()

    def _connect(self):
        try:
            credentials = None
            if self.username and self.password:
                credentials = pika.PlainCredentials(self.username, self.password)

            parameters = pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                virtual_host=self.virtual_host,
                credentials=credentials,
                heartbeat=600, # Keep connection alive
                blocked_connection_timeout=300 # Timeout for blocked connection
            )
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            # Example: Ensure a default exchange if needed, or let publisher specify
            # self.channel.exchange_declare(exchange='default_exchange', exchange_type='direct', durable=True)
            logger.info(f"RabbitMQ Producer connected to {self.host}:{self.port}{self.virtual_host}")
        except pika.exceptions.AMQPConnectionError as e:
            logger.error(f"Failed to connect to RabbitMQ at {self.host}:{self.port}: {e}", exc_info=True)
            self.connection = None
            self.channel = None
            # In a more robust system, you might implement retry logic here
            raise # Re-raise to signal connection failure

    def _ensure_connected(self):
        """Ensures the connection and channel are active, reconnects if necessary."""
        if not self.connection or self.connection.is_closed or \
           not self.channel or self.channel.is_closed:
            logger.warning("RabbitMQ connection lost or not established. Attempting to reconnect...")
            self._connect() # This will raise an exception if reconnection fails

    def publish_message(self, exchange_name: str, routing_key: str, message_body: str,
                        properties: Optional[pika.BasicProperties] = None):
        """
        Publishes a message to the specified exchange with the given routing key.

        Args:
            exchange_name (str): The name of the exchange to publish to.
                                 Use an empty string for the default exchange (direct to queue).
            routing_key (str): The routing key. For default exchange, this is usually the queue name.
            message_body (str): The message body, typically a JSON string.
            properties (Optional[pika.BasicProperties]): Message properties (e.g., delivery_mode).
        """
        self._ensure_connected() # Make sure connection is live
        if not self.channel:
            logger.error("Cannot publish message: RabbitMQ channel is not available.")
            return

        if properties is None:
            # Default to persistent messages
            properties = pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE)

        try:
            # It's good practice for consumers to declare queues and exchanges they need.
            # However, a producer might declare them too for robustness or if it's the first one up.
            # Example: self.channel.queue_declare(queue=routing_key, durable=True)
            # Example: self.channel.exchange_declare(exchange=exchange_name, exchange_type='direct', durable=True)
            # Example: self.channel.queue_bind(exchange=exchange_name, queue=routing_key)

            self.channel.basic_publish(
                exchange=exchange_name,
                routing_key=routing_key,
                body=message_body,
                properties=properties
            )
            logger.debug(f"Message published to exchange '{exchange_name}' with routing key '{routing_key}'. Body: {message_body[:100]}...")
        except pika.exceptions.UnroutableError:
            logger.error(f"Message unroutable: No queue bound to exchange '{exchange_name}' with routing key '{routing_key}'.")
            # Potentially implement dead-lettering or other error handling here.
        except (pika.exceptions.AMQPChannelError, pika.exceptions.AMQPConnectionError) as e:
            logger.error(f"AMQP Error during publish: {e}. Attempting to reconnect and retry once.")
            self.close() # Close broken connection
            try:
                self._connect() # Reconnect
                self.channel.basic_publish( # Retry
                    exchange=exchange_name,
                    routing_key=routing_key,
                    body=message_body,
                    properties=properties
                )
                logger.info("Successfully published message after reconnecting.")
            except Exception as retry_e:
                logger.error(f"Failed to publish message even after retry: {retry_e}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to publish message: {e}", exc_info=True)


    def close(self):
        try:
            if self.channel and self.channel.is_open:
                self.channel.close()
                logger.info("RabbitMQ channel closed.")
            if self.connection and self.connection.is_open:
                self.connection.close()
                logger.info("RabbitMQ connection closed.")
        except Exception as e:
            logger.error(f"Error closing RabbitMQ connection: {e}", exc_info=True)
        finally:
            self.channel = None
            self.connection = None

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Example usage:
    # Ensure RabbitMQ is running (e.g., via `docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management`)

    producer = None
    try:
        # For a local RabbitMQ without specific user/pass (guest/guest usually works if enabled)
        producer = RabbitMQProducer(host='localhost')

        # Before publishing, ensure the queue 'my_test_queue' exists and is bound appropriately
        # if you are using an exchange other than the default.
        # For default exchange, routing_key is the queue name.
        # The consumer should declare the queue. For testing, you can declare it here or via RabbitMQ management UI.
        if producer.channel: # Check if channel was successfully created
             # producer.channel.queue_declare(queue='document_processing_events', durable=True)
             logger.info("Queue 'document_processing_events' should be declared by consumer or pre-exist.")


        test_message = {"id": "doc123", "status": "processed_successfully"}
        producer.publish_message(
            exchange_name='', # Default exchange
            routing_key='document_processing_events',
            message_body=json.dumps(test_message)
        )
        logger.info(f"Test message sent to queue 'document_processing_events'.")

        test_message_2 = {"id": "doc456", "status": "processing_failed", "error": "timeout"}
        producer.publish_message(
            exchange_name='',
            routing_key='document_processing_events',
            message_body=json.dumps(test_message_2)
        )
        logger.info(f"Second test message sent.")

    except Exception as e:
        logger.error(f"Error in RabbitMQProducer example: {e}")
    finally:
        if producer:
            producer.close()
            logger.info("RabbitMQ producer example finished and connection closed.")
