import pytest
from unittest.mock import patch, MagicMock, call
import pika # For pika.exceptions and pika.BasicProperties

# Assuming rabbitmq_producer.py is in Scripts/utils/
from Scripts.utils.rabbitmq_producer import RabbitMQProducer

@pytest.fixture
def mock_pika_connection_params():
    """Mocks pika.ConnectionParameters."""
    with patch("Scripts.utils.rabbitmq_producer.pika.ConnectionParameters") as mock_params:
        yield mock_params

@pytest.fixture
def mock_pika_plain_credentials():
    """Mocks pika.PlainCredentials."""
    with patch("Scripts.utils.rabbitmq_producer.pika.PlainCredentials") as mock_creds:
        yield mock_creds

@pytest.fixture
def mock_pika_blocking_connection():
    """Mocks pika.BlockingConnection and its channel."""
    with patch("Scripts.utils.rabbitmq_producer.pika.BlockingConnection") as mock_conn_constructor:
        mock_connection_instance = MagicMock(spec=pika.BlockingConnection)
        mock_channel_instance = MagicMock(spec=pika.adapters.blocking_connection.BlockingChannel)

        mock_connection_instance.channel.return_value = mock_channel_instance
        mock_connection_instance.is_closed = False # Simulate open connection
        mock_channel_instance.is_closed = False # Simulate open channel

        mock_conn_constructor.return_value = mock_connection_instance
        yield mock_conn_constructor, mock_connection_instance, mock_channel_instance

@pytest.fixture
def rabbitmq_producer(mock_pika_connection_params, mock_pika_plain_credentials, mock_pika_blocking_connection):
    """Provides a RabbitMQProducer instance with mocked pika dependencies."""
    # Reset singleton if any, for clean test state (if producer was a singleton)
    # RabbitMQProducer isn't a singleton in the provided code, so this is fine.
    producer = RabbitMQProducer(host="mockhost", port=5672, username="user", password="password")
    return producer

def test_rabbitmq_producer_init_success(rabbitmq_producer, mock_pika_blocking_connection):
    """Test successful initialization of RabbitMQProducer."""
    _, mock_conn_instance, mock_channel_instance = mock_pika_blocking_connection

    assert rabbitmq_producer.connection == mock_conn_instance
    assert rabbitmq_producer.channel == mock_channel_instance
    mock_conn_instance.channel.assert_called_once()
    # Check if pika.ConnectionParameters was called with correct args (optional, more detailed)

def test_rabbitmq_producer_init_connection_failure(mock_pika_connection_params, mock_pika_plain_credentials, mock_pika_blocking_connection):
    """Test initialization failure if pika.BlockingConnection raises AMQPConnectionError."""
    mock_conn_constructor, _, _ = mock_pika_blocking_connection
    mock_conn_constructor.side_effect = pika.exceptions.AMQPConnectionError("Mock connection error")

    with pytest.raises(pika.exceptions.AMQPConnectionError): # Expecting the error to be re-raised
        RabbitMQProducer(host="mockhost", port=5672)

def test_publish_message_default_exchange(rabbitmq_producer, mock_pika_blocking_connection):
    """Test publishing a message to the default exchange."""
    _, _, mock_channel = mock_pika_blocking_connection

    message_body = '{"key": "value"}'
    routing_key = "test_queue"

    rabbitmq_producer.publish_message(message_body=message_body, routing_key=routing_key)

    mock_channel.basic_publish.assert_called_once()
    args, kwargs = mock_channel.basic_publish.call_args
    assert kwargs['exchange'] == '' # Default exchange
    assert kwargs['routing_key'] == routing_key
    assert kwargs['body'] == message_body
    assert isinstance(kwargs['properties'], pika.BasicProperties)
    assert kwargs['properties'].delivery_mode == pika.spec.PERSISTENT_DELIVERY_MODE

def test_publish_message_to_topic_exchange(rabbitmq_producer, mock_pika_blocking_connection):
    """Test publishing a message to a named topic exchange."""
    _, _, mock_channel = mock_pika_blocking_connection

    message_body = '{"event": "doc.updated"}'
    routing_key = "doc.processed.pdf"
    exchange_name = "document_events_topic"
    exchange_type = "topic"

    rabbitmq_producer.publish_message(
        message_body=message_body,
        routing_key=routing_key,
        exchange_name=exchange_name,
        exchange_type=exchange_type
    )

    mock_channel.exchange_declare.assert_called_once_with(
        exchange=exchange_name, exchange_type=exchange_type, durable=True
    )
    mock_channel.basic_publish.assert_called_once()
    args, kwargs = mock_channel.basic_publish.call_args
    assert kwargs['exchange'] == exchange_name
    assert kwargs['routing_key'] == routing_key
    assert kwargs['body'] == message_body

def test_publish_message_with_custom_properties(rabbitmq_producer, mock_pika_blocking_connection):
    """Test publishing with custom message properties (e.g., priority)."""
    _, _, mock_channel = mock_pika_blocking_connection

    custom_props = pika.BasicProperties(priority=5, delivery_mode=1) # Non-persistent, priority 5

    rabbitmq_producer.publish_message(
        message_body='{"important": true}',
        routing_key="priority_queue",
        properties=custom_props
    )

    mock_channel.basic_publish.assert_called_once()
    args, kwargs = mock_channel.basic_publish.call_args
    assert kwargs['properties'] == custom_props


def test_publish_message_reconnect_and_retry(mock_pika_connection_params, mock_pika_plain_credentials, mock_pika_blocking_connection, caplog):
    """Test that publish attempts to reconnect and retry once if connection is lost."""
    mock_conn_constructor, mock_conn_instance, mock_channel_instance = mock_pika_blocking_connection

    # Initial setup - producer connects successfully
    producer = RabbitMQProducer(host="mockhost", port=5672)
    assert producer.channel == mock_channel_instance

    # Simulate connection error on first publish attempt
    # The first call to basic_publish will raise AMQPConnectionError
    # The second call (after reconnect) should succeed.
    mock_channel_instance.basic_publish.side_effect = [
        pika.exceptions.AMQPConnectionError("Simulated connection error during publish"), # First call fails
        MagicMock()  # Second call succeeds
    ]

    # Simulate connection being closed so _ensure_connected triggers _connect
    mock_conn_instance.is_closed = True
    mock_channel_instance.is_closed = True

    # We need to mock the pika.BlockingConnection again for the reconnect attempt
    # The mock_pika_blocking_connection fixture already patches the constructor.
    # We need to ensure the constructor, when called again by _connect(), returns a new valid mock.

    # Let's refine the fixture or the test for this scenario.
    # For simplicity, let's assume the existing mock_conn_constructor can be called again
    # and will provide a "new" (but same mock) connection/channel that is now "open".

    # To make the mock_conn_instance appear "reconnected" and its channel "re-created":
    new_mock_conn_instance = MagicMock(spec=pika.BlockingConnection)
    new_mock_channel_instance = MagicMock(spec=pika.adapters.blocking_connection.BlockingChannel)
    new_mock_conn_instance.channel.return_value = new_mock_channel_instance
    new_mock_conn_instance.is_closed = False
    new_mock_channel_instance.is_closed = False

    # The constructor needs to return this new instance upon the second call (reconnect)
    # The original mock_conn_instance represents the first connection.
    # The mock_conn_constructor is what pika.BlockingConnection is patched to.
    mock_conn_constructor.side_effect = [mock_conn_instance, new_mock_conn_instance]


    message_body = '{"retry_test": true}'
    routing_key = "retry_q"

    producer.publish_message(message_body=message_body, routing_key=routing_key)

    # Check that basic_publish was called twice (original attempt + retry)
    # The first call was on mock_channel_instance, second on new_mock_channel_instance
    assert mock_channel_instance.basic_publish.call_count == 1
    assert new_mock_channel_instance.basic_publish.call_count == 1

    new_mock_channel_instance.basic_publish.assert_called_with(
        exchange='', routing_key=routing_key, body=message_body, properties=ANY
    )
    assert "Successfully published message after reconnecting." in caplog.text


def test_close_connection(rabbitmq_producer, mock_pika_blocking_connection):
    """Test closing the RabbitMQ connection."""
    _, mock_conn, mock_channel = mock_pika_blocking_connection

    rabbitmq_producer.close()

    mock_channel.close.assert_called_once()
    mock_conn.close.assert_called_once()
    assert rabbitmq_producer.channel is None
    assert rabbitmq_producer.connection is None

def test_ensure_connected_when_already_connected(rabbitmq_producer, mock_pika_blocking_connection):
    """Test _ensure_connected when connection is already fine."""
    _, mock_conn, mock_channel = mock_pika_blocking_connection
    mock_conn.is_closed = False
    mock_channel.is_closed = False

    # Store original channel to check it wasn't re-created
    original_channel_id = id(rabbitmq_producer.channel)

    rabbitmq_producer._ensure_connected() # Should do nothing

    assert id(rabbitmq_producer.channel) == original_channel_id # Channel should be the same
    # No new connection should have been attempted by pika.BlockingConnection
    # mock_pika_blocking_connection[0] is the constructor. It was called once at init.
    assert mock_pika_blocking_connection[0].call_count == 1
