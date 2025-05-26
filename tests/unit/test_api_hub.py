"""
Unit tests for the API Integration Hub routes.
"""
import pytest
import json
import os
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI
import uuid
from datetime import datetime, timedelta

# Import the router to test
from Scripts.api_v1.api_hub_router import router, APIServiceModel, OAuth2AuthorizeModel, \
    OAuth2TokenExchangeModel, APIRequestModel, WebhookModel, BatchRequestModel

# Create a test app
app = FastAPI()
app.include_router(router)
client = TestClient(app)

# Mock services for testing
TEST_SERVICES = [
    {
        "id": "test-api",
        "name": "Test API",
        "description": "API for testing",
        "base_url": "https://api.test.com",
        "auth_type": "oauth2",
        "oauth2_provider": "github",
        "endpoints": [
            {
                "path": "/test",
                "method": "GET",
                "description": "Test endpoint",
                "requires_auth": False
            },
            {
                "path": "/secure",
                "method": "GET",
                "description": "Secure endpoint",
                "requires_auth": True
            }
        ]
    }
]

# Mock webhooks for testing
TEST_WEBHOOKS = [
    {
        "id": "test-webhook",
        "url": "https://webhook.test.com",
        "secret": "test-secret",
        "events": ["service.created", "service.updated"],
        "description": "Test webhook",
        "active": True
    }
]

@pytest.fixture
def mock_services():
    with patch("Scripts.api_v1.api_hub_router.api_services", TEST_SERVICES):
        yield

@pytest.fixture
def mock_webhooks():
    with patch("Scripts.api_v1.api_hub_router.webhooks", TEST_WEBHOOKS):
        yield

@pytest.fixture
def mock_save_services():
    with patch("Scripts.api_v1.api_hub_router.save_api_services", return_value=True) as mock:
        yield mock

@pytest.fixture
def mock_save_webhooks():
    with patch("Scripts.api_v1.api_hub_router.save_webhooks", return_value=True) as mock:
        yield mock

@pytest.fixture
def mock_oauth2_manager():
    mock = MagicMock()
    mock.get_provider.return_value = MagicMock()
    mock.get_provider.return_value.get_authorization_url = MagicMock(return_value="https://auth.test.com")
    mock.validate_jwt_token = MagicMock(return_value={"provider": "github", "external_token": {"access_token": "test-token"}})
    
    with patch("Scripts.api_v1.api_hub_router.oauth2_manager", mock):
        yield mock

@pytest.fixture
def mock_rate_limiter():
    mock = MagicMock()
    mock.check_rate_limit = MagicMock(return_value=True)
    mock.increment_rate_limit = MagicMock()
    mock.get_limit_info = MagicMock(return_value={"limit": 100, "window": 60, "current": 10})
    
    with patch("Scripts.api_v1.api_hub_router.rate_limiter", mock):
        yield mock

@pytest.fixture
def mock_analytics():
    mock = MagicMock()
    mock.track_request = MagicMock()
    mock.track_rate_limit_event = MagicMock()
    mock.track_oauth_event = MagicMock()
    mock.get_request_stats = MagicMock(return_value={
        "total_requests": 100,
        "successful_requests": 90,
        "avg_response_time": 200.5
    })
    mock.get_daily_request_stats = MagicMock(return_value={
        "daily_stats": [
            {
                "date": "2025-05-23",
                "request_count": 50,
                "success_rate": 0.95
            }
        ]
    })
    
    with patch("Scripts.api_v1.api_hub_router.analytics", mock):
        yield mock

@pytest.fixture
def mock_httpx_client():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"success": True, "data": "test data"}
    mock_response.text = '{"success": true, "data": "test data"}'
    
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.put = AsyncMock(return_value=mock_response)
    mock_client.delete = AsyncMock(return_value=mock_response)
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        yield mock_client

# Test service listing
def test_list_services(mock_services):
    """Test listing API services"""
    response = client.get("/api-hub/services")
    assert response.status_code == 200
    assert len(response.json()["services"]) == 1
    assert response.json()["services"][0]["id"] == "test-api"

# Test service details
def test_get_service(mock_services):
    """Test getting API service details"""
    response = client.get("/api-hub/services/test-api")
    assert response.status_code == 200
    assert response.json()["id"] == "test-api"
    
    # Test non-existent service
    response = client.get("/api-hub/services/non-existent")
    assert response.status_code == 404

# Test service registration
def test_register_service(mock_save_services):
    """Test registering a new API service"""
    service_data = {
        "id": "new-api",
        "name": "New API",
        "description": "New API for testing",
        "base_url": "https://api.new.com",
        "auth_type": "none",
        "endpoints": [
            {
                "path": "/test",
                "method": "GET",
                "description": "Test endpoint",
                "requires_auth": False
            }
        ]
    }
    
    with patch("Scripts.api_v1.api_hub_router.notify_webhooks", new_callable=AsyncMock) as mock_notify:
        response = client.post("/api-hub/services", json=service_data)
        assert response.status_code == 200
        assert response.json()["message"] == "API service 'New API' registered successfully"
        assert mock_save_services.called
        # Verify webhook notification was attempted
        mock_notify.assert_called_once()

# Test service update
def test_update_service(mock_services, mock_save_services):
    """Test updating an API service"""
    service_data = {
        "id": "test-api",
        "name": "Updated Test API",
        "description": "Updated API for testing",
        "base_url": "https://api.test.com",
        "auth_type": "oauth2",
        "oauth2_provider": "github",
        "endpoints": [
            {
                "path": "/test",
                "method": "GET",
                "description": "Test endpoint",
                "requires_auth": False
            }
        ]
    }
    
    with patch("Scripts.api_v1.api_hub_router.notify_webhooks", new_callable=AsyncMock) as mock_notify:
        response = client.put("/api-hub/services/test-api", json=service_data)
        assert response.status_code == 200
        assert response.json()["message"] == "API service 'Updated Test API' updated successfully"
        assert mock_save_services.called
        # Verify webhook notification was attempted
        mock_notify.assert_called_once()

# Test service deletion
def test_delete_service(mock_services, mock_save_services):
    """Test deleting an API service"""
    with patch("Scripts.api_v1.api_hub_router.notify_webhooks", new_callable=AsyncMock) as mock_notify:
        response = client.delete("/api-hub/services/test-api")
        assert response.status_code == 200
        assert "deleted successfully" in response.json()["message"]
        assert mock_save_services.called
        # Verify webhook notification was attempted
        mock_notify.assert_called_once()

# Test OAuth2 authorization
def test_oauth2_authorize(mock_oauth2_manager, mock_analytics):
    """Test getting OAuth2 authorization URL"""
    auth_data = {
        "provider": "github",
        "redirect_uri": "https://example.com/callback",
        "scope": "user repo",
        "state": "test-state"
    }
    
    response = client.post("/api-hub/oauth2/authorize", json=auth_data)
    assert response.status_code == 200
    assert response.json()["authorization_url"] == "https://auth.test.com"
    assert response.json()["provider"] == "github"
    assert response.json()["state"] == "test-state"
    assert mock_analytics.track_oauth_event.called

# Test OAuth2 token exchange
@pytest.mark.asyncio
async def test_oauth2_token(mock_oauth2_manager, mock_analytics):
    """Test exchanging OAuth2 code for token"""
    # Mock the exchange_code_for_token method
    mock_oauth2_manager.get_provider.return_value.exchange_code_for_token = AsyncMock(
        return_value={"access_token": "test-token"}
    )
    mock_oauth2_manager.create_jwt_token = MagicMock(return_value="jwt-token")
    
    token_data = {
        "provider": "github",
        "code": "test-code",
        "redirect_uri": "https://example.com/callback"
    }
    
    response = client.post("/api-hub/oauth2/token", json=token_data)
    assert response.status_code == 200
    assert response.json()["access_token"] == "jwt-token"
    assert response.json()["provider"] == "github"
    assert mock_analytics.track_oauth_event.called

# Test API request
@pytest.mark.asyncio
async def test_make_api_request(mock_services, mock_rate_limiter, mock_analytics, mock_httpx_client):
    """Test making an API request"""
    request_data = {
        "service_id": "test-api",
        "endpoint": "/test",
        "method": "GET",
        "params": {"param1": "value1"}
    }
    
    response = client.post("/api-hub/request", json=request_data)
    assert response.status_code == 200
    assert "result" in response.json()
    assert mock_rate_limiter.check_rate_limit.called
    assert mock_rate_limiter.increment_rate_limit.called
    assert mock_analytics.track_request.called
    assert mock_httpx_client.get.called

# Test rate limit endpoint
def test_get_rate_limits(mock_rate_limiter):
    """Test getting rate limit status"""
    response = client.get("/api-hub/rate-limits")
    assert response.status_code == 200
    assert "limits" in response.json()

# Test webhook management
def test_webhook_management(mock_webhooks, mock_save_webhooks):
    """Test webhook CRUD operations"""
    # List webhooks
    response = client.get("/api-hub/webhooks")
    assert response.status_code == 200
    assert len(response.json()["webhooks"]) == 1
    
    # Get webhook
    response = client.get("/api-hub/webhooks/test-webhook")
    assert response.status_code == 200
    assert response.json()["id"] == "test-webhook"
    
    # Create webhook
    webhook_data = {
        "url": "https://new-webhook.test.com",
        "events": ["service.deleted"],
        "description": "New test webhook"
    }
    
    with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
        with patch("Scripts.api_v1.api_hub_router.notify_webhooks", new_callable=AsyncMock) as mock_notify:
            response = client.post("/api-hub/webhooks", json=webhook_data)
            assert response.status_code == 200
            assert "webhook" in response.json()
            assert mock_save_webhooks.called
            mock_notify.assert_called_once()
    
    # Update webhook
    update_data = {
        "id": "test-webhook",
        "url": "https://updated-webhook.test.com",
        "events": ["service.created"],
        "active": True
    }
    
    with patch("Scripts.api_v1.api_hub_router.notify_webhooks", new_callable=AsyncMock) as mock_notify:
        response = client.put("/api-hub/webhooks/test-webhook", json=update_data)
        assert response.status_code == 200
        assert "webhook" in response.json()
        assert mock_save_webhooks.called
        mock_notify.assert_called_once()
    
    # Delete webhook
    with patch("Scripts.api_v1.api_hub_router.notify_webhooks", new_callable=AsyncMock) as mock_notify:
        response = client.delete("/api-hub/webhooks/test-webhook")
        assert response.status_code == 200
        assert "deleted successfully" in response.json()["message"]
        assert mock_save_webhooks.called
        mock_notify.assert_called_once()

# Test analytics endpoints
def test_analytics_endpoints(mock_analytics):
    """Test analytics endpoints"""
    # Request analytics
    response = client.get("/api-hub/analytics/requests")
    assert response.status_code == 200
    assert mock_analytics.get_request_stats.called
    
    # Daily analytics
    response = client.get("/api-hub/analytics/dashboard")
    assert response.status_code == 200
    assert mock_analytics.get_request_stats.called
    assert mock_analytics.get_daily_request_stats.called

# Test service health check
@pytest.mark.asyncio
async def test_service_health_check(mock_services, mock_analytics, mock_httpx_client):
    """Test service health check"""
    response = client.post("/api-hub/services/test-api/health-check")
    assert response.status_code == 200
    assert response.json()["service_id"] == "test-api"
    assert response.json()["is_healthy"] is True
    assert mock_analytics.track_request.called
    assert mock_httpx_client.get.called

# Test batch requests
@pytest.mark.asyncio
async def test_batch_requests(mock_services, mock_rate_limiter, mock_analytics, mock_httpx_client):
    """Test batch API requests"""
    batch_data = {
        "requests": [
            {
                "service_id": "test-api",
                "endpoint": "/test",
                "method": "GET"
            },
            {
                "service_id": "test-api",
                "endpoint": "/test",
                "method": "POST",
                "body": {"key": "value"}
            }
        ],
        "parallel": True
    }
    
    response = client.post("/api-hub/batch", json=batch_data)
    assert response.status_code == 200
    assert response.json()["batch_size"] == 2
    assert response.json()["successful_requests"] == 2
    assert mock_rate_limiter.check_rate_limit.called
    assert mock_rate_limiter.increment_rate_limit.called

# Test test webhook endpoint
@pytest.mark.asyncio
async def test_webhook_test(mock_httpx_client):
    """Test the webhook test endpoint"""
    test_data = {
        "url": "https://webhook-test.com/endpoint",
        "secret": "test-secret"
    }
    
    with patch("Scripts.api_v1.api_hub_router.send_webhook", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = {"success": True, "status_code": 200}
        response = client.post("/api-hub/webhooks/test", json=test_data)
        assert response.status_code == 200
        assert "Test webhook sent successfully" in response.json()["message"]
        mock_send.assert_called_once()
