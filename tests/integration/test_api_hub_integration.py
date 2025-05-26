"""
Integration tests for the API Integration Hub.
Tests the integration between the API hub and other system components.
"""
import pytest
import asyncio
import json
import os
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
import httpx
from datetime import datetime, timedelta

from Scripts.rag_api import app
from Scripts.api_v1.api_hub_router import APIServiceModel, WebhookModel

client = TestClient(app)

# Test with real database connections but mock external services
@pytest.fixture
def test_db_path():
    """Create a temporary test database path"""
    db_path = "data/test_api_hub/test_analytics.db"
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    yield db_path
    # Clean up the test database after tests
    if os.path.exists(db_path):
        os.remove(db_path)

@pytest.fixture
def setup_test_api_files():
    """Set up test API service and webhook files"""
    services_file = "data/test_api_hub/services.json"
    webhooks_file = "data/test_api_hub/webhooks.json"
    
    os.makedirs(os.path.dirname(services_file), exist_ok=True)
    
    # Create test services
    test_services = [
        {
            "id": "test-github-api",
            "name": "Test GitHub API",
            "description": "GitHub API for testing",
            "base_url": "https://api.github.com",
            "auth_type": "oauth2",
            "oauth2_provider": "github",
            "endpoints": [
                {
                    "path": "/user",
                    "method": "GET",
                    "description": "Get user info",
                    "requires_auth": True
                },
                {
                    "path": "/repos/{owner}/{repo}",
                    "method": "GET",
                    "description": "Get repo info",
                    "requires_auth": False
                }
            ]
        }
    ]
    
    # Create test webhooks
    test_webhooks = [
        {
            "id": "test-integration-webhook",
            "url": "https://webhook.test.integration.com",
            "events": ["service.created", "request.error"],
            "description": "Test integration webhook",
            "active": True
        }
    ]
    
    # Write test files
    with open(services_file, 'w') as f:
        json.dump(test_services, f)
    
    with open(webhooks_file, 'w') as f:
        json.dump(test_webhooks, f)
    
    # Patch file paths
    with patch("Scripts.api_v1.api_hub_router.API_SERVICES_FILE", services_file):
        with patch("Scripts.api_v1.api_hub_router.WEBHOOKS_FILE", webhooks_file):
            yield services_file, webhooks_file
    
    # Clean up after tests
    os.remove(services_file)
    os.remove(webhooks_file)

@pytest.fixture
def setup_analytics_db(test_db_path):
    """Set up a test analytics database"""
    with patch("Scripts.api_hub.analytics.ApiUsageAnalytics.__init__", return_value=None):
        with patch("Scripts.api_hub.analytics.ApiUsageAnalytics.db_path", test_db_path):
            with patch("Scripts.api_hub.analytics.ApiUsageAnalytics._ensure_db_exists"):
                yield

@pytest.fixture
def mock_httpx():
    """Mock httpx for external API calls"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"success": True, "data": "test data"}
    mock_response.text = '{"success": true, "data": "test data"}'
    mock_response.raise_for_status = MagicMock()
    
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.put = AsyncMock(return_value=mock_response)
    mock_client.delete = AsyncMock(return_value=mock_response)
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        yield mock_client

@pytest.mark.integration
def test_api_hub_integration_with_main_app(setup_test_api_files, mock_httpx):
    """Test that the API hub router is properly integrated with the main app"""
    response = client.get("/api/v1/api-hub/services")
    assert response.status_code == 200
    assert len(response.json()["services"]) == 1
    assert response.json()["services"][0]["id"] == "test-github-api"

@pytest.mark.integration
def test_api_hub_oauth2_integration(setup_test_api_files, mock_httpx, setup_analytics_db):
    """Test OAuth2 integration"""
    # Mock OAuth2 manager
    with patch("Scripts.api_hub.oauth2.OAuth2Manager.get_provider") as mock_get_provider:
        mock_provider = MagicMock()
        mock_provider.get_authorization_url = MagicMock(return_value="https://auth.test.com")
        mock_get_provider.return_value = mock_provider
        
        response = client.post("/api/v1/api-hub/oauth2/authorize", json={
            "provider": "github",
            "redirect_uri": "https://example.com/callback",
            "scope": "user repo",
            "state": "test-state"
        })
        
        assert response.status_code == 200
        assert response.json()["authorization_url"] == "https://auth.test.com"
        assert mock_get_provider.called

@pytest.mark.integration
def test_api_hub_rate_limiting_integration(setup_test_api_files, mock_httpx, setup_analytics_db):
    """Test rate limiting integration"""
    # Make a series of requests to test rate limiting
    with patch("Scripts.api_hub.rate_limiter.RateLimiter.check_rate_limit") as mock_check:
        with patch("Scripts.api_hub.rate_limiter.RateLimiter.increment_rate_limit") as mock_increment:
            # First request succeeds
            mock_check.return_value = True
            
            response = client.post("/api/v1/api-hub/request", json={
                "service_id": "test-github-api",
                "endpoint": "/repos/{owner}/{repo}",
                "method": "GET",
                "params": {
                    "owner": "testorg",
                    "repo": "testrepo"
                }
            })
            
            assert response.status_code == 200
            assert mock_check.called
            assert mock_increment.called
            
            # Second request fails due to rate limit
            mock_check.return_value = False
            
            response = client.post("/api/v1/api-hub/request", json={
                "service_id": "test-github-api",
                "endpoint": "/repos/{owner}/{repo}",
                "method": "GET",
                "params": {
                    "owner": "testorg",
                    "repo": "testrepo"
                }
            })
            
            assert response.status_code == 429
            assert "Rate limit exceeded" in response.json()["detail"]

@pytest.mark.integration
def test_api_hub_analytics_integration(setup_test_api_files, mock_httpx, setup_analytics_db):
    """Test analytics integration"""
    with patch("Scripts.api_hub.analytics.ApiUsageAnalytics.track_request") as mock_track:
        with patch("Scripts.api_hub.analytics.ApiUsageAnalytics.get_request_stats") as mock_get_stats:
            # Set up mock stats
            mock_get_stats.return_value = {
                "total_requests": 100,
                "successful_requests": 90,
                "failed_requests": 10,
                "avg_response_time": 150.5,
                "service_counts": {"test-github-api": 100},
                "status_code_counts": {"200": 90, "404": 5, "500": 5}
            }
            
            # Test tracking a request
            response = client.post("/api/v1/api-hub/request", json={
                "service_id": "test-github-api",
                "endpoint": "/repos/{owner}/{repo}",
                "method": "GET",
                "params": {
                    "owner": "testorg",
                    "repo": "testrepo"
                }
            })
            
            assert response.status_code == 200
            assert mock_track.called
            
            # Test getting analytics
            response = client.get("/api/v1/api-hub/analytics/requests")
            assert response.status_code == 200
            assert mock_get_stats.called

@pytest.mark.integration
def test_api_hub_webhook_integration(setup_test_api_files, mock_httpx):
    """Test webhook integration"""
    with patch("Scripts.api_v1.api_hub_router.send_webhook", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = {"success": True, "status_code": 200}
        
        # Create a new service that should trigger a webhook
        service_data = {
            "id": "new-test-api",
            "name": "New Test API",
            "description": "API for testing webhooks",
            "base_url": "https://api.newtest.com",
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
        
        response = client.post("/api/v1/api-hub/services", json=service_data)
        assert response.status_code == 200
        
        # Give the async task a chance to execute
        import asyncio
        import time
        time.sleep(0.1)  # Small delay to allow async task to run
        
        # Check if webhook notification was sent
        assert mock_send.called
        
        # Verify the webhook was called with the right event
        call_args = mock_send.call_args
        assert call_args is not None
        webhook, event_type, payload = call_args[0]
        assert event_type == "service.created"
        assert payload["service"]["id"] == "new-test-api"

@pytest.mark.integration
def test_batch_request_integration(setup_test_api_files, mock_httpx, setup_analytics_db):
    """Test batch request functionality"""
    with patch("Scripts.api_hub.rate_limiter.RateLimiter.check_rate_limit", return_value=True):
        batch_data = {
            "requests": [
                {
                    "service_id": "test-github-api",
                    "endpoint": "/repos/{owner}/{repo}",
                    "method": "GET",
                    "params": {
                        "owner": "testorg",
                        "repo": "repo1"
                    }
                },
                {
                    "service_id": "test-github-api",
                    "endpoint": "/repos/{owner}/{repo}",
                    "method": "GET",
                    "params": {
                        "owner": "testorg",
                        "repo": "repo2"
                    }
                }
            ],
            "parallel": True
        }
        
        response = client.post("/api/v1/api-hub/batch", json=batch_data)
        assert response.status_code == 200
        assert response.json()["batch_size"] == 2
        assert response.json()["successful_requests"] == 2
        
        # Test with one failing request
        with patch("httpx.AsyncClient.__aenter__") as mock_enter:
            mock_client = AsyncMock()
            
            # Create success and error responses
            success_response = MagicMock()
            success_response.status_code = 200
            success_response.json.return_value = {"success": True}
            success_response.raise_for_status = MagicMock()
            
            error_response = MagicMock()
            error_response.status_code = 404
            error_response.json.return_value = {"error": "Not found"}
            error_response.text = '{"error": "Not found"}'
            error_response.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError(
                "Not found", request=MagicMock(), response=error_response
            ))
            
            # Set up client to return different responses
            mock_client.get = AsyncMock(side_effect=[success_response, error_response])
            mock_enter.return_value = mock_client
            
            response = client.post("/api/v1/api-hub/batch", json=batch_data)
            assert response.status_code == 200
            result = response.json()
            assert result["batch_size"] == 2
            assert result["successful_requests"] == 1
            assert result["failed_requests"] == 1

@pytest.mark.integration
def test_service_health_check_integration(setup_test_api_files, mock_httpx, setup_analytics_db):
    """Test service health check integration"""
    with patch("Scripts.api_hub.analytics.ApiUsageAnalytics.track_request") as mock_track:
        response = client.post("/api/v1/api-hub/services/test-github-api/health-check")
        assert response.status_code == 200
        assert response.json()["service_id"] == "test-github-api"
        assert response.json()["is_healthy"] is True
        assert mock_track.called
        assert mock_httpx.get.called
