import pika
import os
import logging
import time
import json

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
        logger.error(f"Critical: Failed to import get_config_value for RabbitMQ Producer. Error: {e}")
        def get_config_value(env_var_name, yaml_path=None, default=None): # Basic fallback
            return os.getenv(env_var_name, default)

# RabbitMQ Configuration
RABBITMQ_HOST = get_config_value("RABBITMQ_HOST", yaml_path="message_queue.rabbitmq.host", default="localhost")
RABBITMQ_PORT = int(get_config_value("RABBITMQ_PORT", yaml_path="message_queue.rabbitmq.port", default=5672))
RABBITMQ_USER = get_config_value("RABBITMQ_USER", yaml_path="message_queue.rabbitmq.user", default="user")
RABBITMQ_PASS = get_config_value("RABBITMQ_PASSWORD", yaml_path="message_queue.rabbitmq.password", default="password")
# For K8s, RABBITMQ_HOST would be "rabbitmq-service" (the service name)

# Example queue and exchange names
EXAMPLE_EXCHANGE_NAME = 'example_exchange'
EXAMPLE_QUEUE_NAME = 'example_task_queue'
EXAMPLE_ROUTING_KEY = 'example.task'


def send_message(message_body: Union[str, Dict], num_messages: int = 1):
    connection = None
    try:
        credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
        parameters = pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            credentials=credentials,
            # virtual_host='/' # Default vhost
            # heartbeat=600, # Example: longer heartbeat
            # blocked_connection_timeout=300
        )
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()

        # Declare a durable exchange (optional, good practice for routing)
        # channel.exchange_declare(exchange=EXAMPLE_EXCHANGE_NAME, exchange_type='direct', durable=True)

        # Declare a durable queue
        # If you bind this queue to an exchange, messages sent to the exchange with the right routing key will go here.
        # For direct-to-queue publishing (default exchange), just declare queue.
        channel.queue_declare(queue=EXAMPLE_QUEUE_NAME, durable=True) # Durable ensures queue survives broker restart

        # If using a custom exchange, bind the queue to it:
        # channel.queue_bind(exchange=EXAMPLE_EXCHANGE_NAME, queue=EXAMPLE_QUEUE_NAME, routing_key=EXAMPLE_ROUTING_KEY)

        logger.info(f"Connected to RabbitMQ at {RABBITMQ_HOST}:{RABBITMQ_PORT}. Sending {num_messages} message(s)...")

        for i in range(num_messages):
            current_message_body = message_body
            if isinstance(message_body, dict):
                # If dict, add a sequence number for multiple messages
                if num_messages > 1:
                    current_message_body = message_body.copy()
                    current_message_body['sequence'] = i + 1
                message_payload = json.dumps(current_message_body)
            else: # string
                if num_messages > 1:
                    message_payload = f"{current_message_body} - Message #{i+1}"
                else:
                    message_payload = current_message_body

            channel.basic_publish(
                exchange='', # Publish to the default exchange (empty string)
                routing_key=EXAMPLE_QUEUE_NAME, # Route directly to the queue
                # If using custom exchange:
                # exchange=EXAMPLE_EXCHANGE_NAME,
                # routing_key=EXAMPLE_ROUTING_KEY,
                body=message_payload.encode('utf-8'),
                properties=pika.BasicProperties(
                    delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE # Make message persistent
                )
            )
            logger.info(f" [x] Sent: '{message_payload[:100]}...'")
            if num_messages > 1 and i < num_messages -1 : # Add a small delay if sending multiple
                time.sleep(0.1)

    except pika.exceptions.AMQPConnectionError as e:
        logger.error(f"Failed to connect to RabbitMQ at {RABBITMQ_HOST}:{RABBITMQ_PORT}. Error: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        if connection and connection.is_open:
            try:
                connection.close()
                logger.info("RabbitMQ connection closed.")
            except Exception as e_close:
                logger.error(f"Error closing RabbitMQ connection: {e_close}")

if __name__ == '__main__':
    # Example: Send a JSON message
    sample_json_message = {
        "task_id": f"task_{int(time.time())}",
        "payload": "Process this item.",
        "timestamp": datetime.utcnow().isoformat(),
        "priority": "high"
    }
    send_message(sample_json_message)

    # Example: Send a simple string message multiple times
    # send_message("Hello RabbitMQ from Python Producer!", num_messages=3)

    # To run this producer:
    # 1. Ensure RabbitMQ server is running and accessible.
    #    (e.g., via Docker: `docker run -d --hostname my-rabbit --name some-rabbit -p 5672:5672 -p 15672:15672 -e RABBITMQ_DEFAULT_USER=user -e RABBITMQ_DEFAULT_PASS=password rabbitmq:3-management`)
    #    Or deployed in Kubernetes using the provided manifests.
    # 2. Set environment variables if defaults are not suitable:
    #    export RABBITMQ_HOST="your_rabbitmq_host" # e.g., localhost or rabbitmq-service
    #    export RABBITMQ_PORT="5672"
    #    export RABBITMQ_USER="user"
    #    export RABBITMQ_PASSWORD="password"
    # 3. Run the script: python Scripts/utils/rabbitmq_producer_example.py
    #
    # You can then check the RabbitMQ Management UI (if using `rabbitmq:3-management` image,
    # usually at http://<rabbitmq_host>:15672) to see the message in the 'example_task_queue'.
    # Default credentials for UI if not overridden: guest/guest (but we set user/password in Docker example).
    # For K8s deployment, use port-forwarding to access management UI:
    # kubectl port-forward service/rabbitmq-service 15672:15672
    # Then access http://localhost:15672 with user/password from K8s deployment.
