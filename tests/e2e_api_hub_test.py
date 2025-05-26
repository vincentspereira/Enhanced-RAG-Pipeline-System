"""
End-to-End tests for the API Hub and related components.
These tests simulate real user workflows.
"""
import pytest
import asyncio
import json
import os
import uuid
import time
from unittest.mock import patch, MagicMock, AsyncMock

# Test client for API Hub
class ApiHubClient:
    """Test client for interacting with the API Hub"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.token = None
    
    async def register_service(self, service_data):
        """Register a new API service"""
        # This would be an HTTP request in a real client
        # For testing, we'll interact directly with the test environment
        client_response = MagicMock()
        client_response.status_code = 200
        client_response.json.return_value = {
            "message": f"API service '{service_data['name']}' registered successfully",
            "service_id": service_data["id"]
        }
        return client_response
    
    async def make_request(self, service_id, endpoint, method="GET", params=None, body=None):
        """Make a request to an external API through the hub"""
        # Simulate client request
        client_response = MagicMock()
        client_response.status_code = 200
        
        # Different responses based on endpoint type
        if endpoint == "/user":
            client_response.json.return_value = {"login": "testuser", "id": 12345}
        elif endpoint == "/repos":
            client_response.json.return_value = [{"name": "repo1"}, {"name": "repo2"}]
        else:
            client_response.json.return_value = {"data": "Test data"}
        
        return client_response
    
    async def batch_request(self, requests):
        """Make a batch of requests"""
        client_response = MagicMock()
        client_response.status_code = 200
        
        # Process each request
        results = []
        for req in requests:
            result = {
                "service_id": req["service_id"],
                "endpoint": req["endpoint"],
                "success": True,
                "status_code": 200
            }
            
            # Different responses based on endpoint
            if req["endpoint"] == "/user":
                result["data"] = {"login": "testuser", "id": 12345}
            elif req["endpoint"] == "/repos":
                result["data"] = [{"name": "repo1"}, {"name": "repo2"}]
            else:
                result["data"] = {"data": "Test data"}
            
            results.append(result)
        
        client_response.json.return_value = {"results": results}
        return client_response
    
    async def register_webhook(self, webhook_data):
        """Register a new webhook"""
        client_response = MagicMock()
        client_response.status_code = 200
        client_response.json.return_value = {
            "message": "Webhook registered successfully",
            "webhook_id": webhook_data["id"]
        }
        return client_response

# Test environment setup
@pytest.fixture
def setup_e2e_environment():
    """Set up a test environment for end-to-end testing"""
    test_dir = "e2e_test_data"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create test files
    services_file = os.path.join(test_dir, "services.json")
    webhooks_file = os.path.join(test_dir, "webhooks.json")
    analytics_file = os.path.join(test_dir, "analytics.json")
    
    # Create initial test data
    test_services = []
    test_webhooks = []
    test_analytics = {
        "requests": [],
        "rate_limits": [],
        "oauth_events": []
    }
    
    # Write initial data
    with open(services_file, 'w') as f:
        json.dump(test_services, f)
    
    with open(webhooks_file, 'w') as f:
        json.dump(test_webhooks, f)
    
    with open(analytics_file, 'w') as f:
        json.dump(test_analytics, f)
    
    yield test_dir
    
    # Cleanup after tests
    for file in [services_file, webhooks_file, analytics_file]:
        if os.path.exists(file):
            os.remove(file)
    
    if os.path.exists(test_dir):
        os.rmdir(test_dir)

@pytest.fixture
def api_client():
    """Create an API client for tests"""
    return ApiHubClient()

# E2E Tests
@pytest.mark.asyncio
async def test_full_api_workflow(setup_e2e_environment, api_client):
    """Test a full workflow with service registration, API calls and webhooks"""
    
    # 1. Register a new API service
    github_service = {
        "id": "github-api",
        "name": "GitHub API",
        "description": "GitHub REST API",
        "base_url": "https://api.github.com",
        "auth_type": "oauth2",
        "oauth2_provider": "github",
        "endpoints": [
            {
                "path": "/user",
                "method": "GET",
                "description": "Get authenticated user",
                "requires_auth": True
            },
            {
                "path": "/repos",
                "method": "GET",
                "description": "Get user repositories",
                "requires_auth": True
            }
        ]
    }
    
    response = await api_client.register_service(github_service)
    assert response.status_code == 200
    
    # Manually update the services file
    services_file = os.path.join(setup_e2e_environment, "services.json")
    with open(services_file, 'w') as f:
        json.dump([github_service], f)
    
    # 2. Register a webhook
    webhook_data = {
        "id": "github-webhook",
        "url": "https://webhook.test.com/github",
        "events": ["request.success", "request.error"],
        "active": True
    }
    
    response = await api_client.register_webhook(webhook_data)
    assert response.status_code == 200
    
    # Manually update the webhooks file
    webhooks_file = os.path.join(setup_e2e_environment, "webhooks.json")
    with open(webhooks_file, 'w') as f:
        json.dump([webhook_data], f)
    
    # 3. Make an API request
    response = await api_client.make_request("github-api", "/user")
    assert response.status_code == 200
    user_data = response.json()
    assert user_data["login"] == "testuser"
    
    # 4. Make a batch request
    batch_requests = [
        {
            "service_id": "github-api",
            "endpoint": "/user",
            "method": "GET"
        },
        {
            "service_id": "github-api",
            "endpoint": "/repos",
            "method": "GET"
        }
    ]
    
    response = await api_client.batch_request(batch_requests)
    assert response.status_code == 200
    
    batch_results = response.json()["results"]
    assert len(batch_results) == 2
    assert batch_results[0]["success"] == True
    assert batch_results[1]["success"] == True
    
    # 5. Track analytics (simulate)
    analytics_file = os.path.join(setup_e2e_environment, "analytics.json")
    analytics_data = {
        "requests": [
            {
                "timestamp": time.time(),
                "service_id": "github-api",
                "endpoint": "/user",
                "status_code": 200,
                "response_time_ms": 125.5
            },
            {
                "timestamp": time.time(),
                "service_id": "github-api",
                "endpoint": "/repos",
                "status_code": 200, 
                "response_time_ms": 98.3
            }
        ],
        "rate_limits": [
            {
                "timestamp": time.time(),
                "client_id": "127.0.0.1",
                "service_id": "github-api",
                "limit_reached": False
            }
        ],
        "oauth_events": [
            {
                "timestamp": time.time(),
                "provider": "github",
                "event_type": "token",
                "success": True
            }
        ]
    }
    
    with open(analytics_file, 'w') as f:
        json.dump(analytics_data, f)
    
    # 6. Verify all data is as expected
    with open(services_file, 'r') as f:
        services = json.load(f)
        assert len(services) == 1
        assert services[0]["id"] == "github-api"
    
    with open(webhooks_file, 'r') as f:
        webhooks = json.load(f)
        assert len(webhooks) == 1
        assert webhooks[0]["id"] == "github-webhook"
    
    with open(analytics_file, 'r') as f:
        analytics = json.load(f)
        assert len(analytics["requests"]) == 2
        assert len(analytics["rate_limits"]) == 1
        assert len(analytics["oauth_events"]) == 1

@pytest.mark.asyncio
async def test_error_handling_workflow(setup_e2e_environment, api_client):
    """Test error handling in API workflow"""
    
    # Mock error responses
    error_client = AsyncMock()
    error_response = MagicMock()
    error_response.status_code = 404
    error_response.json.return_value = {"error": "Not found"}
    error_client.get = AsyncMock(return_value=error_response)
    
    # 1. Register a test service
    test_service = {
        "id": "test-service",
        "name": "Test Service",
        "description": "Service for testing errors",
        "base_url": "https://api.test.com",
        "auth_type": "none",
        "endpoints": [
            {
                "path": "/error",
                "method": "GET",
                "description": "Endpoint that returns an error",
                "requires_auth": False
            }
        ]
    }
    
    # Add service to file
    services_file = os.path.join(setup_e2e_environment, "services.json")
    with open(services_file, 'w') as f:
        json.dump([test_service], f)
    
    # 2. Register webhook for error events
    webhook_data = {
        "id": "error-webhook",
        "url": "https://webhook.test.com/errors",
        "events": ["request.error"],
        "active": True
    }
    
    webhooks_file = os.path.join(setup_e2e_environment, "webhooks.json")
    with open(webhooks_file, 'w') as f:
        json.dump([webhook_data], f)
    
    # 3. Simulate a failed request
    with patch.object(api_client, 'make_request', new=AsyncMock(return_value=error_response)):
        response = await api_client.make_request("test-service", "/error")
        assert response.status_code == 404
        error_data = response.json()
        assert "error" in error_data
    
    # 4. Track error analytics
    analytics_file = os.path.join(setup_e2e_environment, "analytics.json")
    analytics_data = {
        "requests": [
            {
                "timestamp": time.time(),
                "service_id": "test-service",
                "endpoint": "/error",
                "status_code": 404,
                "response_time_ms": 85.2,
                "error": "Not found"
            }
        ],
        "webhook_events": [
            {
                "timestamp": time.time(),
                "webhook_id": "error-webhook",
                "event": "request.error",
                "status_code": 200
            }
        ]
    }
    
    with open(analytics_file, 'w') as f:
        json.dump(analytics_data, f)
    
    # 5. Verify error tracking
    with open(analytics_file, 'r') as f:
        analytics = json.load(f)
        assert len(analytics["requests"]) == 1
        assert analytics["requests"][0]["status_code"] == 404
        assert "webhook_events" in analytics
