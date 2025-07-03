import pytest
from unittest.mock import patch, MagicMock, ANY
import pika # For pika.exceptions and pika.spec
import json
import os

# Assuming document_event_consumer.py is in Scripts/workers/
from Scripts.workers.document_event_consumer import DocumentEventConsumer, QUEUE_NAME as DEFAULT_QUEUE_NAME

# Mock os.getenv for RabbitMQ connection parameters
@pytest.fixture(autouse=True)
def mock_rabbitmq_env_vars(mocker):
    env_map = {
        "RABBITMQ_HOST": "mockhost",
        "RABBITMQ_PORT": "5672", # os.getenv returns string
        "RABBITMQ_USER": "mockuser",
        "RABBITMQ_PASSWORD": "mockpassword",
        "RABBITMQ_VHOST": "/",
    }
    mocker.patch("Scripts.workers.document_event_consumer.os.getenv",
                 side_effect=lambda key, default=None: env_map.get(key, default))

@pytest.fixture
def mock_pika_blocking_connection_consumer():
    """Mocks pika.BlockingConnection for consumer tests."""
    with patch("Scripts.workers.document_event_consumer.pika.BlockingConnection") as mock_conn_constructor:
        mock_connection_instance = MagicMock(spec=pika.BlockingConnection)
        mock_channel_instance = MagicMock(spec=pika.adapters.blocking_connection.BlockingChannel)

        mock_connection_instance.channel.return_value = mock_channel_instance
        mock_connection_instance.is_closed = False
        mock_channel_instance.is_closed = False

        mock_conn_constructor.return_value = mock_connection_instance
        yield mock_conn_constructor, mock_connection_instance, mock_channel_instance

@pytest.fixture
def document_consumer(mock_rabbitmq_env_vars, mock_pika_blocking_connection_consumer):
    """Provides a DocumentEventConsumer instance with mocked pika dependencies."""
    consumer = DocumentEventConsumer(
        host="mockhost", # These will be overridden by mock_rabbitmq_env_vars if it patches os.getenv correctly
        port=5672,
        username="mockuser",
        password="mockpassword",
        virtual_host="/",
        queue_name="test_doc_events_queue"
    )
    return consumer

def test_consumer_connect_and_setup_queues(document_consumer, mock_pika_blocking_connection_consumer):
    """Test that the consumer connects and declares exchange, queue, and binding correctly."""
    _, mock_conn, mock_channel = mock_pika_blocking_connection_consumer

    # _connect is called during DocumentEventConsumer instantiation

    # Verify exchange declaration
    mock_channel.exchange_declare.assert_called_once_with(
        exchange='document_events_exchange', exchange_type='topic', durable=True
    )
    # Verify queue declaration
    mock_channel.queue_declare.assert_called_once_with(
        queue=document_consumer.queue_name, durable=True, arguments={} # Assuming default empty args
    )
    # Verify queue binding
    mock_channel.queue_bind.assert_called_once_with(
        exchange='document_events_exchange', queue=document_consumer.queue_name, routing_key='doc.processed.*'
    )

def test_message_callback_success_document_processed(document_consumer, mock_pika_blocking_connection_consumer, caplog):
    """Test the _message_callback for a valid 'document_processed' event."""
    _, _, mock_channel = mock_pika_blocking_connection_consumer

    mock_method = MagicMock(spec=pika.spec.Basic.Deliver)
    mock_method.delivery_tag = 12345
    mock_method.routing_key = "doc.processed.pdf" # Example routing key

    mock_properties = MagicMock(spec=pika.spec.BasicProperties)
    mock_properties.priority = 1 # Example priority

    message_data = {
        "event_type": "document_processed",
        "parent_document_id": "doc-uuid-5678",
        "filename": "report.pdf",
        "chunks_indexed": 10,
        "status": "indexed_chunked"
    }
    message_body_bytes = json.dumps(message_data).encode('utf-8')

    document_consumer._message_callback(mock_channel, mock_method, mock_properties, message_body_bytes)

    assert f"Received message (Priority: 1, RoutingKey: doc.processed.pdf): {json.dumps(message_data)}" in caplog.text
    assert f"Processing 'document_processed' event: ID=doc-uuid-5678, File=report.pdf, Chunks=10, Status=indexed_chunked" in caplog.text
    mock_channel.basic_ack.assert_called_once_with(delivery_tag=mock_method.delivery_tag)

def test_message_callback_unknown_event_type(document_consumer, mock_pika_blocking_connection_consumer, caplog):
    """Test _message_callback with an unknown event type."""
    _, _, mock_channel = mock_pika_blocking_connection_consumer
    mock_method = MagicMock(spec=pika.spec.Basic.Deliver, delivery_tag=1)
    mock_method.routing_key = "some.other.key"
    message_data = {"event_type": "some_other_event", "data": "payload"}
    message_body_bytes = json.dumps(message_data).encode('utf-8')

    document_consumer._message_callback(mock_channel, mock_method, None, message_body_bytes)

    assert "Received unknown event_type: 'some_other_event'" in caplog.text
    mock_channel.basic_ack.assert_called_once_with(delivery_tag=1) # Still acks if processing logic handles unknown type gracefully

def test_message_callback_json_decode_error(document_consumer, mock_pika_blocking_connection_consumer, caplog):
    """Test _message_callback with a message that is not valid JSON."""
    _, _, mock_channel = mock_pika_blocking_connection_consumer
    mock_method = MagicMock(spec=pika.spec.Basic.Deliver, delivery_tag=2)
    invalid_json_body_bytes = b"this is not json"

    document_consumer._message_callback(mock_channel, mock_method, None, invalid_json_body_bytes)

    assert "Failed to decode JSON message: this is not json" in caplog.text
    mock_channel.basic_nack.assert_called_once_with(delivery_tag=2, requeue=False)
    mock_channel.basic_ack.assert_not_called()

def test_message_callback_processing_exception(document_consumer, mock_pika_blocking_connection_consumer, caplog):
    """Test _message_callback when an exception occurs during message processing logic."""
    _, _, mock_channel = mock_pika_blocking_connection_consumer
    mock_method = MagicMock(spec=pika.spec.Basic.Deliver, delivery_tag=3)
    message_data = {"event_type": "document_processed"} # Valid JSON
    message_body_bytes = json.dumps(message_data).encode('utf-8')

    # Patch a part of the processing logic to raise an error
    with patch.object(logging.getLogger("Scripts.workers.document_event_consumer"), "info") as mock_logger_info:
        # Make one of the log calls (or any part of the processing) raise an error
        # For example, if the logger itself fails for some reason during processing
        mock_logger_info.side_effect = Exception("Simulated processing error")

        document_consumer._message_callback(mock_channel, mock_method, None, message_body_bytes)

    assert "Error processing message: Simulated processing error" in caplog.text
    mock_channel.basic_nack.assert_called_once_with(delivery_tag=3, requeue=False)
    mock_channel.basic_ack.assert_not_called()

@patch("Scripts.workers.document_event_consumer.time.sleep") # Mock time.sleep to speed up test
def test_start_consuming_handles_connection_error_and_retries(mock_sleep, mock_pika_blocking_connection_consumer, caplog):
    """Test that start_consuming attempts to reconnect after AMQPConnectionError."""
    mock_conn_constructor, mock_conn_instance, mock_channel_instance = mock_pika_blocking_connection_consumer

    # First connection attempt (during __init__) is mocked to succeed by the fixture.
    consumer = DocumentEventConsumer("h","p","u","p","vh","q")

    # Simulate connection error when start_consuming calls _connect or channel operations
    # Let _connect succeed initially, but then channel.start_consuming raises error
    # Forcing an error that would lead to reconnection loop.
    # The consumer's start_consuming loop will call _connect, then basic_qos, basic_consume, start_consuming.
    # Let's make start_consuming itself fail to simulate a lost connection during consumption.

    # First call to start_consuming will raise error, second will "succeed" (by breaking loop for test)
    mock_channel_instance.start_consuming.side_effect = [
        pika.exceptions.AMQPConnectionError("Simulated connection error during consumption"),
        KeyboardInterrupt("Stop test loop") # Stop after one "successful" reconnect and consume start
    ]

    # Ensure _connect (which is called by constructor and then again in loop) returns new valid mocks on retry
    # The mock_pika_blocking_connection_consumer fixture sets up the first successful connection.
    # We need to ensure subsequent calls to pika.BlockingConnection() inside _connect() also get mocks.
    # The fixture already patches pika.BlockingConnection, so it will return the same mock_conn_instance.
    # We need to ensure its state is "reset" or it behaves as a new connection.

    # For the reconnect, mock_conn_instance.channel() should be called again.
    # And then queue_declare, exchange_declare, queue_bind on the new channel.

    # Let's reset call counts on the channel mock before start_consuming
    mock_channel_instance.reset_mock()

    with pytest.raises(KeyboardInterrupt): # Expect test to stop due to this
        consumer.start_consuming()

    assert "RabbitMQ connection error: Simulated connection error during consumption. Retrying in 10 seconds..." in caplog.text
    # Check if setup methods were called again after the "reconnect"
    # The constructor calls them once. The loop calls _connect, which calls them again.
    assert mock_channel_instance.exchange_declare.call_count >= 2 # Initial + at least one retry
    assert mock_channel_instance.queue_declare.call_count >= 2
    assert mock_channel_instance.queue_bind.call_count >= 2
    assert mock_channel_instance.basic_qos.call_count >= 1 # Called before start_consuming
    assert mock_channel_instance.basic_consume.call_count >= 1
    mock_sleep.assert_any_call(10) # Check that it tried to sleep before retrying

# Note: Testing the full loop of start_consuming with KeyboardInterrupt is a bit complex.
# More granular tests on _connect and the loop's error handling branches might be added if needed.
# The above test verifies that a connection error leads to a retry attempt.
