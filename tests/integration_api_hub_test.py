"""
Integration tests for the API Hub with mock external services.
"""
import pytest
import asyncio
import json
import os
from unittest.mock import patch, MagicMock, AsyncMock

# Create a test directory for data
@pytest.fixture
def setup_test_environment():
    """Set up a test environment with files and directories"""
    test_dir = "test_data"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create test files
    services_file = os.path.join(test_dir, "services.json")
    webhooks_file = os.path.join(test_dir, "webhooks.json")
    
    # Create initial test data
    test_services = [
        {
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
                }
            ]
        }
    ]
    
    test_webhooks = [
        {
            "id": "test-webhook",
            "url": "https://webhook.test.com",
            "events": ["service.created", "service.updated"],
            "active": True
        }
    ]
    
    # Write initial data
    with open(services_file, 'w') as f:
        json.dump(test_services, f)
    
    with open(webhooks_file, 'w') as f:
        json.dump(test_webhooks, f)
    
    yield test_dir
    
    # Cleanup after tests
    if os.path.exists(services_file):
        os.remove(services_file)
    
    if os.path.exists(webhooks_file):
        os.remove(webhooks_file)
    
    if os.path.exists(test_dir):
        os.rmdir(test_dir)

def test_service_registration_and_webhook(setup_test_environment):
    """Test service registration and webhook notification integration"""
    
    # Mock HTTP client for webhook notifications
    mock_response = MagicMock()
    mock_response.status_code = 200
    
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post = AsyncMock(return_value=mock_response)
    
    # Load services from file
    services_file = os.path.join(setup_test_environment, "services.json")
    with open(services_file, 'r') as f:
        services = json.load(f)
    
    # Load webhooks from file
    webhooks_file = os.path.join(setup_test_environment, "webhooks.json")
    with open(webhooks_file, 'r') as f:
        webhooks = json.load(f)
    
    # Register new service
    new_service = {
        "id": "new-test-api",
        "name": "New Test API",
        "description": "New API for testing",
        "base_url": "https://api.new.test.com",
        "auth_type": "api_key",
        "endpoints": [
            {
                "path": "/data",
                "method": "GET",
                "description": "Get data",
                "requires_auth": True
            }
        ]
    }
    
    services.append(new_service)
    
    # Save services
    with open(services_file, 'w') as f:
        json.dump(services, f)
    
    # Simulate webhook notification
    async def notify_webhooks(event, data):
        # Filter webhooks for this event
        matching_webhooks = [w for w in webhooks if event in w["events"] and w["active"]]
        
        for webhook in matching_webhooks:
            async with mock_client as client:
                await client.post(
                    webhook["url"],
                    json={
                        "event": event,
                        "data": data
                    }
                )
        
        return len(matching_webhooks)
    
    # Run webhook notification
    with patch('httpx.AsyncClient', return_value=mock_client):
        loop = asyncio.get_event_loop()
        webhook_count = loop.run_until_complete(
            notify_webhooks("service.created", {"service": new_service})
        )
        
        # Verify webhook was notified
        assert webhook_count == 1
        assert mock_client.post.called
    
    # Verify service was saved
    with open(services_file, 'r') as f:
        updated_services = json.load(f)
        assert len(updated_services) == 2
        assert any(s["id"] == "new-test-api" for s in updated_services)

def test_batch_request_integration(setup_test_environment):
    """Test batch request integration with mock external services"""
    
    # Mock responses for different endpoints
    mock_responses = {
        "/user": {"login": "testuser", "id": 12345},
        "/repos": [{"name": "repo1"}, {"name": "repo2"}]
    }
    
    # Mock HTTP client
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    
    def mock_get(url, **kwargs):
        for endpoint, response_data in mock_responses.items():
            if endpoint in url:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = response_data
                return mock_response
        
        # Default response
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": "Not found"}
        return mock_response
    
    mock_client.get = AsyncMock(side_effect=mock_get)
    
    # Batch request model
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
    
    # Load services
    services_file = os.path.join(setup_test_environment, "services.json")
    with open(services_file, 'r') as f:
        services = json.load(f)
    
    # Function to process batch requests
    async def process_batch_requests(requests):
        results = []
        service_map = {s["id"]: s for s in services}
        
        async with mock_client as client:
            for req in requests:
                service = service_map.get(req["service_id"])
                if not service:
                    results.append({
                        "success": False,
                        "error": f"Service '{req['service_id']}' not found"
                    })
                    continue
                
                base_url = service["base_url"]
                url = f"{base_url.rstrip('/')}/{req['endpoint'].lstrip('/')}"
                
                try:
                    response = await client.get(url)
                    results.append({
                        "success": response.status_code == 200,
                        "status_code": response.status_code,
                        "data": response.json()
                    })
                except Exception as e:
                    results.append({
                        "success": False,
                        "error": str(e)
                    })
        
        return {"results": results}
    
    # Run batch processing
    with patch('httpx.AsyncClient', return_value=mock_client):
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(process_batch_requests(batch_requests))
        
        # Verify results
        assert len(result["results"]) == 2
        assert result["results"][0]["success"] == True
        assert result["results"][0]["data"]["login"] == "testuser"
        assert result["results"][1]["success"] == True
        assert len(result["results"][1]["data"]) == 2
