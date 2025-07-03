# Message Queueing with RabbitMQ

This document outlines the use of RabbitMQ for asynchronous message queueing within the RAG system.

## 1. Overview

RabbitMQ is used to decouple services and handle tasks asynchronously. For example, after a document is processed and indexed, an event can be published to a message queue. Other services or workers can then consume these events to perform follow-up actions like notifications, further analytics, or triggering other workflows.

## 2. Setup and Configuration

*   **Deployment:** RabbitMQ can be deployed as part of the system using the provided Helm chart (`charts/rag-system/`), which includes RabbitMQ as an optional subchart. Alternatively, an existing RabbitMQ instance can be used.
*   **Connection Parameters:** Services connect to RabbitMQ using parameters typically configured via environment variables or `config.yaml`:
    *   `RABBITMQ_HOST`
    *   `RABBITMQ_PORT` (default: 5672)
    *   `RABBITMQ_USER` (optional)
    *   `RABBITMQ_PASSWORD` (optional)
    *   `RABBITMQ_VHOST` (default: `/`)
*   **Client Library:** The system uses the `pika` Python library for RabbitMQ integration.

## 3. Core Concepts Used

### Exchanges

*   **Topic Exchange (`document_events_exchange`):**
    *   The primary exchange used for document-related events is named `document_events_exchange` and is of type `topic`.
    *   This allows for flexible routing of messages based on routing key patterns.
    *   Producers (like `DocProcessingService`) declare this exchange to ensure it exists.
    *   **Declaration (Server-side or by Producer/Consumer):**
        ```python
        # In Pika (Python client)
        channel.exchange_declare(exchange='document_events_exchange', exchange_type='topic', durable=True)
        ```

### Queues

*   **Consumer Queues:** Consumers declare their own queues and bind them to an exchange with specific routing key patterns.
    *   Example queue: `doc_processing_log_queue` (declared by `DocumentEventConsumer`).
    *   Queues are declared as `durable=True` to survive RabbitMQ restarts (messages will also be persisted if published as persistent).
    *   **Declaration (Server-side or by Consumer):**
        ```python
        # In Pika
        channel.queue_declare(queue='my_consumer_queue', durable=True)
        ```

### Routing and Binding Keys

*   **Routing Keys (Producer):** When publishing a message to a topic exchange, the producer specifies a routing key.
    *   Example: `doc.processed.pdf`, `doc.processed.txt`, `user.action.updated_profile`
    *   The `DocProcessingService` uses routing keys like `doc.processed.<file_extension>`.
*   **Binding Keys (Consumer):** Consumers bind their queues to the topic exchange using binding keys, which can include wildcards:
    *   `*` (star) can substitute for exactly one word.
    *   `#` (hash) can substitute for zero or more words.
    *   Example: A consumer interested in all processed documents might bind its queue with `doc.processed.*`.
    *   Example: A consumer interested only in PDF processing events might use `doc.processed.pdf`.
    *   **Binding (Consumer):**
        ```python
        # In Pika
        channel.queue_bind(exchange='document_events_exchange', queue='my_consumer_queue', routing_key='doc.processed.*')
        ```

## 4. Advanced Features (Conceptual Setup)

The following features are primarily configured on the RabbitMQ server itself, often by an administrator or through deployment scripts (e.g., RabbitMQ Operator, Helm chart values for the RabbitMQ subchart if it supports these directly). Client applications (producer/consumer) can provide hints or set properties that interact with these server-side configurations.

### Priority Queues

*   **Purpose:** Allows messages to be processed in an order determined by their priority, rather than strictly FIFO. Useful for ensuring high-priority tasks are handled sooner.
*   **Server-Side Setup:**
    1.  Declare a queue with the `x-max-priority` argument. This defines the maximum priority value the queue will support (e.g., 1 to 10).
        ```
        # RabbitMQ Management UI or rabbitmqctl
        # arguments: {"x-max-priority": 10}
        ```
*   **Client-Side (Producer):**
    *   When publishing a message, set the `priority` property in `pika.BasicProperties`.
        ```python
        # In Pika
        properties = pika.BasicProperties(
            delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE,
            priority=5 # Integer value, up to x-max-priority
        )
        channel.basic_publish(..., properties=properties)
        ```
    *   Higher numbers typically mean higher priority.
*   **Client-Side (Consumer):** No special configuration needed for the consumer to respect priorities from a priority queue; RabbitMQ handles delivery order.

### Dead-Letter Exchanges (DLX) and Queues (DLQ)

*   **Purpose:** To handle messages that cannot be processed successfully by a consumer. Instead of being endlessly re-queued or discarded, they are sent to a DLX and then typically routed to a DLQ for inspection and potential manual intervention or reprocessing.
*   **Server-Side Setup:**
    1.  **Declare a DLX:** This is a normal exchange (e.g., a direct or fanout exchange) where dead-lettered messages will be sent.
        ```
        # rabbitmqctl or Management UI
        # Example: declare exchange name=my_app_dlx type=direct
        ```
    2.  **Declare a DLQ:** This is a normal queue where messages from the DLX will be routed.
        ```
        # rabbitmqctl or Management UI
        # Example: declare queue name=my_app_dlq durable=true
        ```
    3.  **Bind DLQ to DLX:**
        ```
        # rabbitmqctl or Management UI
        # Example: bind queue my_app_dlq to exchange my_app_dlx (with a routing key if DLX is not fanout)
        ```
    4.  **Configure Main Queue to Use DLX:** When declaring the main consumer queue, set the `x-dead-letter-exchange` argument to the name of your DLX. Optionally, set `x-dead-letter-routing-key` if the DLX requires a specific routing key to route to the DLQ.
        ```
        # In Pika (Consumer's queue declaration)
        queue_arguments = {
            'x-dead-letter-exchange': 'my_app_dlx',
            # 'x-dead-letter-routing-key': 'dlq_processing_failures' # Optional
        }
        channel.queue_declare(queue='main_app_queue', durable=True, arguments=queue_arguments)
        ```
*   **Client-Side (Consumer):**
    *   When a consumer negatively acknowledges (`basic_nack` or `basic_reject`) a message with `requeue=False`, RabbitMQ will check the queue's DLX configuration. If configured, the message is sent to the DLX.
    *   Reasons for dead-lettering:
        *   Message explicitly rejected with `requeue=False`.
        *   Message TTL expires.
        *   Queue length limit exceeded.

## 5. Current Implementation Notes

*   `Scripts/utils/rabbitmq_producer.py`:
    *   The `publish_message` method now supports specifying `exchange_name` and `exchange_type`.
    *   It declares the exchange (if not default) to make it more robust.
    *   Includes a comment placeholder for setting `pika.BasicProperties(priority=...)`.
*   `Scripts/services/doc_processing_service.py`:
    *   Publishes messages to a topic exchange named `document_events_exchange`.
    *   Uses routing keys like `doc.processed.<file_extension>`.
*   `Scripts/workers/document_event_consumer.py`:
    *   Declares the `document_events_exchange` (topic type).
    *   Declares its own consumer queue (e.g., `document_processing_log_queue`).
    *   Binds its queue to the exchange using a pattern like `doc.processed.*`.
    *   Includes comments regarding `x-max-priority` and `x-dead-letter-exchange` arguments for queue declaration.
    *   Logs message priority if present in properties.

Further enhancements to fully utilize priority and DLQs would involve more detailed server-side RabbitMQ configuration, which can be managed via its administration tools, CLI, or through infrastructure-as-code if deploying RabbitMQ via Helm/Operator with such capabilities.
