import pika
import logging
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)

class RabbitMQConnection:
    def __init__(self, host: str, port: int, username: Optional[str], password: Optional[str], virtual_host: str = "/"):
        self.host = host
        self.port = port
        self.virtual_host = virtual_host
        self.credentials = None
        if username and password:
            self.credentials = pika.PlainCredentials(username, password)

        self.connection: Optional[pika.BlockingConnection] = None
        self.channel: Optional[pika.channel.Channel] = None

    def connect(self):
        """Establishes a connection to RabbitMQ and opens a channel."""
        if self.connection and self.connection.is_open and self.channel and self.channel.is_open:
            logger.debug("RabbitMQ connection and channel already open.")
            return
        try:
            params = pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                virtual_host=self.virtual_host,
                credentials=self.credentials
            )
            self.connection = pika.BlockingConnection(params)
            self.channel = self.connection.channel()
            logger.info(f"Successfully connected to RabbitMQ at {self.host}:{self.port}{self.virtual_host}")
        except pika.exceptions.AMQPConnectionError as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}", exc_info=True)
            self.connection = None
            self.channel = None
            raise # Re-raise the exception so callers can handle it

    def close(self):
        """Closes the RabbitMQ channel and connection."""
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

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def declare_queue(self, queue_name: str, durable: bool = True):
        """
        Declares a queue.

        Args:
            queue_name (str): The name of the queue.
            durable (bool): If True, the queue will survive broker restarts.
        """
        if not self.channel or not self.channel.is_open:
            logger.error("Cannot declare queue: RabbitMQ channel is not open.")
            raise ConnectionError("RabbitMQ channel is not open. Call connect() first.")

        try:
            self.channel.queue_declare(queue=queue_name, durable=durable)
            logger.info(f"Queue '{queue_name}' declared (durable={durable}).")
        except Exception as e:
            logger.error(f"Failed to declare queue '{queue_name}': {e}", exc_info=True)
            raise

    def publish_message(
        self,
        queue_name: str,
        message_body: str,
        exchange: str = '', # Default exchange for direct-to-queue publishing
        properties: Optional[pika.BasicProperties] = None
    ):
        """
        Publishes a message to the specified queue.

        Args:
            queue_name (str): The name of the queue (used as routing_key for default exchange).
            message_body (str): The message body to publish.
            exchange (str): The exchange to publish to. Defaults to the default exchange.
            properties (Optional[pika.BasicProperties]): Message properties (e.g., delivery_mode).
        """
        if not self.channel or not self.channel.is_open:
            logger.error("Cannot publish message: RabbitMQ channel is not open.")
            raise ConnectionError("RabbitMQ channel is not open. Call connect() first.")

        if properties is None:
            # Make messages persistent by default if queue is durable
            properties = pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE)

        try:
            self.channel.basic_publish(
                exchange=exchange,
                routing_key=queue_name, # For default exchange, routing_key is the queue name
                body=message_body.encode('utf-8'), # Ensure body is bytes
                properties=properties
            )
            logger.info(f"Message published to queue '{queue_name}'. Body: '{message_body[:100]}...'")
        except Exception as e:
            logger.error(f"Failed to publish message to queue '{queue_name}': {e}", exc_info=True)
            raise

    def consume_messages(
        self,
        queue_name: str,
        callback_function: Callable[[pika.channel.Channel, pika.spec.Basic.Deliver, pika.spec.BasicProperties, bytes], Any],
        auto_ack: bool = False # Manual acknowledgement is safer
    ):
        """
        Starts consuming messages from the specified queue. This is a blocking operation.

        Args:
            queue_name (str): The name of the queue to consume from.
            callback_function (Callable): The function to call when a message is received.
                                         It should accept: ch, method, properties, body.
            auto_ack (bool): If True, messages are acknowledged automatically.
                             If False (recommended), callback_function must call ch.basic_ack().
        """
        if not self.channel or not self.channel.is_open:
            logger.error("Cannot consume messages: RabbitMQ channel is not open.")
            raise ConnectionError("RabbitMQ channel is not open. Call connect() first.")

        try:
            self.channel.basic_qos(prefetch_count=1) # Process one message at a time per consumer
            self.channel.basic_consume(
                queue=queue_name,
                on_message_callback=callback_function,
                auto_ack=auto_ack
            )
            logger.info(f"Starting to consume messages from queue '{queue_name}'. Waiting for messages...")
            self.channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("Consumer stopped by user (KeyboardInterrupt).")
            self.close()
        except Exception as e:
            logger.error(f"Error during message consumption from queue '{queue_name}': {e}", exc_info=True)
            self.close() # Ensure connection is closed on error
            raise

# Example of getting a configured RabbitMQConnection instance
# This would typically be managed by the application (e.g., in enhanced_api.py app.state or via dependency injection)
_rabbitmq_connection_instance: Optional[RabbitMQConnection] = None

def initialize_rabbitmq_connection(config: Any): # config should be RabbitMQConfig compatible
    global _rabbitmq_connection_instance
    if _rabbitmq_connection_instance is not None:
        logger.warning("RabbitMQ connection already initialized.")
        return

    try:
        _rabbitmq_connection_instance = RabbitMQConnection(
            host=config.host,
            port=config.port,
            username=config.username, # These should come from secrets in a real app
            password=config.password, # These should come from secrets in a real app
            virtual_host=config.virtual_host
        )
        # Optionally connect here or let it connect on first use / with context manager
        # _rabbitmq_connection_instance.connect()
        logger.info("RabbitMQConnection object initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize RabbitMQConnection object: {e}", exc_info=True)
        _rabbitmq_connection_instance = None


def get_rabbitmq_connection() -> Optional[RabbitMQConnection]:
    if _rabbitmq_connection_instance is None:
        logger.error("RabbitMQ connection has not been initialized. Call initialize_rabbitmq_connection() first.")
        # raise RuntimeError("RabbitMQ connection not initialized") # Or handle by returning None
        return None
    return _rabbitmq_connection_instance

if __name__ == '__main__':
    # This is example usage and would require RabbitMQ running.
    # In a real app, config would come from a config manager.

    logging.basicConfig(level=logging.INFO) # For standalone test

    class MockRabbitMQConfig:
        host = "localhost"
        port = 5672
        username = "raguser"
        password = "ragpassword"
        virtual_host = "/"
        default_document_queue = "test_doc_queue"

    test_config = MockRabbitMQConfig()
    initialize_rabbitmq_connection(test_config)

    mq_conn = get_rabbitmq_connection()

    if mq_conn:
        try:
            with mq_conn: # Uses context manager to connect/close
                mq_conn.declare_queue(test_config.default_document_queue, durable=True)

                for i in range(3):
                    msg_body = f"Hello RabbitMQ! Message {i}"
                    mq_conn.publish_message(
                        queue_name=test_config.default_document_queue,
                        message_body=msg_body
                    )

                logger.info("Messages published. To test consumption, run a consumer script.")

                # Example of how a consumer might be started (blocking, so usually in a separate process/thread)
                # def sample_callback(ch, method, properties, body):
                #     print(f" [x] Received {body.decode()}")
                #     ch.basic_ack(delivery_tag=method.delivery_tag) # Acknowledge message
                #
                # print(" [*] Waiting for messages. To exit press CTRL+C")
                # try:
                #     mq_conn.consume_messages(test_config.default_document_queue, sample_callback)
                # except Exception as e:
                #     print(f"Consumer error: {e}")

        except ConnectionError as ce:
            logger.error(f"Could not connect to RabbitMQ for example: {ce}")
        except Exception as e:
            logger.error(f"An error occurred in RabbitMQ example: {e}", exc_info=True)
    else:
        logger.error("Failed to get RabbitMQ connection for example.")
