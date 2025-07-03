import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import httpx

# Assuming n8n_client.py is in Scripts/utils/
from Scripts.utils.n8n_client import trigger_n8n_workflow

@pytest.mark.asyncio
async def test_trigger_n8n_workflow_success_json_response():
    """Test successful n8n workflow trigger with JSON response."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "workflow_triggered", "execution_id": "123"}
    mock_response.text = '{"status": "workflow_triggered", "execution_id": "123"}'
    mock_response.raise_for_status = MagicMock() # Does nothing if status is OK

    # Patch AsyncClient context manager behavior
    mock_async_client = AsyncMock(spec=httpx.AsyncClient)
    mock_async_client.__aenter__.return_value = mock_async_client # Important for 'async with'
    mock_async_client.__aexit__.return_value = None
    mock_async_client.post = AsyncMock(return_value=mock_response)

    with patch("Scripts.utils.n8n_client.httpx.AsyncClient", return_value=mock_async_client) as mock_httpx_constructor:
        webhook_url = "http://mock-n8n/webhook/test"
        payload = {"key": "value"}

        result = await trigger_n8n_workflow(webhook_url, payload)

        mock_httpx_constructor.assert_called_once() # Check AsyncClient was initiated
        mock_async_client.post.assert_called_once_with(webhook_url, json=payload, timeout=30)
        mock_response.raise_for_status.assert_called_once()
        assert result == {"status": "workflow_triggered", "execution_id": "123"}

@pytest.mark.asyncio
async def test_trigger_n8n_workflow_success_text_response():
    """Test successful n8n workflow trigger with non-JSON text response."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.side_effect = ValueError("Not JSON") # Simulate non-JSON response
    mock_response.text = "Workflow started successfully"
    mock_response.raise_for_status = MagicMock()

    mock_async_client = AsyncMock(spec=httpx.AsyncClient)
    mock_async_client.__aenter__.return_value = mock_async_client
    mock_async_client.__aexit__.return_value = None
    mock_async_client.post = AsyncMock(return_value=mock_response)

    with patch("Scripts.utils.n8n_client.httpx.AsyncClient", return_value=mock_async_client):
        webhook_url = "http://mock-n8n/webhook/test"
        payload = {"key": "value"}

        result = await trigger_n8n_workflow(webhook_url, payload)

        assert result == {"status": "success", "content": "Workflow started successfully"}

@pytest.mark.asyncio
async def test_trigger_n8n_workflow_http_status_error(caplog):
    """Test n8n workflow trigger with HTTPStatusError (e.g., 4xx or 5xx)."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 404
    mock_response.text = "Webhook not found"
    # raise_for_status should be called and raise an error
    mock_response.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError(
        "Not Found", request=MagicMock(), response=mock_response
    ))

    mock_async_client = AsyncMock(spec=httpx.AsyncClient)
    mock_async_client.__aenter__.return_value = mock_async_client
    mock_async_client.__aexit__.return_value = None
    mock_async_client.post = AsyncMock(return_value=mock_response)

    with patch("Scripts.utils.n8n_client.httpx.AsyncClient", return_value=mock_async_client):
        webhook_url = "http://mock-n8n/webhook/invalid"
        payload = {"key": "value"}

        result = await trigger_n8n_workflow(webhook_url, payload)

        assert result is None
        mock_response.raise_for_status.assert_called_once()
        assert "HTTP error triggering n8n workflow" in caplog.text
        assert "404 - Webhook not found" in caplog.text


@pytest.mark.asyncio
async def test_trigger_n8n_workflow_request_error(caplog):
    """Test n8n workflow trigger with a generic RequestError (e.g., connection timeout)."""
    mock_async_client = AsyncMock(spec=httpx.AsyncClient)
    mock_async_client.__aenter__.return_value = mock_async_client
    mock_async_client.__aexit__.return_value = None
    mock_async_client.post = AsyncMock(side_effect=httpx.RequestError("Connection timed out", request=MagicMock()))

    with patch("Scripts.utils.n8n_client.httpx.AsyncClient", return_value=mock_async_client):
        webhook_url = "http://mock-n8n/webhook/timeout"
        payload = {"key": "value"}

        result = await trigger_n8n_workflow(webhook_url, payload)

        assert result is None
        assert "Request error triggering n8n workflow" in caplog.text
        assert "Connection timed out" in caplog.text

@pytest.mark.asyncio
async def test_trigger_n8n_workflow_unexpected_error(caplog):
    """Test n8n workflow trigger with an unexpected error during the process."""
    mock_async_client = AsyncMock(spec=httpx.AsyncClient)
    mock_async_client.__aenter__.return_value = mock_async_client
    mock_async_client.__aexit__.return_value = None
    # Simulate an error after a successful post but before/during response processing
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.side_effect = Exception("Unexpected parsing issue") # Error during .json()

    mock_async_client.post = AsyncMock(return_value=mock_response)

    with patch("Scripts.utils.n8n_client.httpx.AsyncClient", return_value=mock_async_client):
        webhook_url = "http://mock-n8n/webhook/unexpected"
        payload = {"key": "value"}

        result = await trigger_n8n_workflow(webhook_url, payload)

        assert result is None
        assert "Unexpected error triggering n8n workflow" in caplog.text
        assert "Unexpected parsing issue" in caplog.text

@pytest.mark.asyncio
async def test_trigger_n8n_workflow_custom_timeout():
    """Test that custom timeout is passed to httpx client."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "ok"}
    mock_response.raise_for_status = MagicMock()

    mock_async_client = AsyncMock(spec=httpx.AsyncClient)
    mock_async_client.__aenter__.return_value = mock_async_client
    mock_async_client.__aexit__.return_value = None
    mock_async_client.post = AsyncMock(return_value=mock_response)

    with patch("Scripts.utils.n8n_client.httpx.AsyncClient", return_value=mock_async_client):
        webhook_url = "http://mock-n8n/webhook/custom_timeout"
        payload = {"key": "value"}
        custom_timeout = 15

        await trigger_n8n_workflow(webhook_url, payload, timeout=custom_timeout)

        mock_async_client.post.assert_called_once_with(webhook_url, json=payload, timeout=custom_timeout)

# To run these tests:
# Ensure pytest and pytest-asyncio are installed.
# Run from the root directory: pytest tests/utils/test_n8n_client.py
