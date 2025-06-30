import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock, patch # AsyncMock not strictly needed if PMS is sync

from Scripts.enhanced_api import app # App from enhanced_api to get TestClient with full setup
from Scripts.llm.prompt_system import PromptManagementSystem, PromptTemplate as ActualPromptTemplate
# ActualPromptTemplate for creating test data
from Scripts.auth.auth_manager import AuthUser, Role, Permission # For creating mock users

# Use fixtures from global conftest.py for TestClient and tokens
# test_app_client: provides a TestClient with mocked RAGPipeline, but real AuthManager via startup
# admin_token: provides a valid JWT for an admin user
# regular_user_token: provides a valid JWT for a regular user

@pytest.fixture
def mock_pms():
    """Fixture to mock PromptManagementSystem."""
    pms_instance = MagicMock(spec=PromptManagementSystem)
    pms_instance.library = MagicMock()
    pms_instance.library.templates = {} # Simulate empty library initially
    pms_instance.library.get_template = MagicMock(side_effect=lambda name: pms_instance.library.templates.get(name))
    pms_instance.library.add_template = MagicMock(return_value=True)
    pms_instance.library.delete_template = MagicMock(return_value=True)

    pms_instance.create_template = MagicMock(side_effect=lambda template, name, description, variables: ActualPromptTemplate(template=template, name=name, description=description, variables=variables or []))
    pms_instance.update_template = MagicMock(side_effect=lambda name, template, description, variables: ActualPromptTemplate(template=template or "updated", name=name, description=description, variables=variables or []))

    pms_instance.get_usage_stats = MagicMock(return_value={"total_usage": 0, "templates": []})
    pms_instance.get_performance_metrics = MagicMock(return_value={"total_metrics_count": 0, "templates": []})

    # For merged endpoints
    # Assuming PromptManagementSystem now has (or delegates to a manager that has) these:
    pms_instance.manager = MagicMock() # Simulate internal manager if PMS delegates
    pms_instance.manager.get_template_versions = MagicMock(return_value=[])
    pms_instance.manager.record_feedback = MagicMock(return_value=True)

    # If PMS has these methods directly:
    # pms_instance.get_template_versions = MagicMock(return_value=[])
    # pms_instance.record_feedback = MagicMock(return_value=True)

    return pms_instance

@pytest.fixture(autouse=True)
def override_dependencies(mock_pms):
    """Override dependencies for prompt_router for all tests in this module."""
    # This overrides get_pms for the scope of tests in this file
    app.dependency_overrides[__import__('Scripts.api_v1.prompt_router', fromlist=['get_pms']).get_pms] = lambda: mock_pms
    # The get_current_user dependency is handled by AuthManager in app.state via TestClient
    yield
    app.dependency_overrides = {} # Clear overrides after tests


# --- Test Cases ---

def test_list_templates_unauthenticated(test_app_client: TestClient):
    response = test_app_client.get("/api/v1/prompts/templates")
    assert response.status_code == 401 # Expecting unauthorized

def test_list_templates_authenticated(test_app_client: TestClient, regular_user_token: str, mock_pms: MagicMock):
    mock_template = ActualPromptTemplate(name="test_template", template="Hello {name}", variables=["name"], version="1.1")
    mock_pms.library.templates = {"test_template": mock_template}

    response = test_app_client.get(
        "/api/v1/prompts/templates",
        headers={"Authorization": f"Bearer {regular_user_token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["name"] == "test_template"
    assert data[0]["version"] == "1.1"

def test_create_template_authenticated(test_app_client: TestClient, admin_token: str, mock_pms: MagicMock):
    template_data = {"name": "new_prompt", "template": "Test {var}", "variables": ["var"], "description": "A test prompt"}

    # Mock the create_template method on pms to return an object that can be dict-ed
    created_mock_template = ActualPromptTemplate(name=template_data["name"], template=template_data["template"], variables=template_data["variables"], description=template_data["description"], version="1.0")
    mock_pms.create_template.return_value = created_mock_template

    response = test_app_client.post(
        "/api/v1/prompts/templates",
        json=template_data,
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200, response.text
    mock_pms.create_template.assert_called_once_with(
        name=template_data["name"],
        template=template_data["template"],
        description=template_data["description"],
        variables=template_data["variables"]
    )
    assert response.json()["name"] == "new_prompt"

def test_get_template_authenticated(test_app_client: TestClient, regular_user_token: str, mock_pms: MagicMock):
    mock_template = ActualPromptTemplate(name="existing_template", template="Content", variables=[], version="1.0")
    mock_pms.library.templates = {"existing_template": mock_template}

    response = test_app_client.get(
        "/api/v1/prompts/templates/existing_template",
        headers={"Authorization": f"Bearer {regular_user_token}"}
    )
    assert response.status_code == 200
    assert response.json()["name"] == "existing_template"

def test_get_template_not_found(test_app_client: TestClient, regular_user_token: str, mock_pms: MagicMock):
    mock_pms.library.get_template.return_value = None # Explicitly mock not found
    response = test_app_client.get(
        "/api/v1/prompts/templates/nonexistent_template",
        headers={"Authorization": f"Bearer {regular_user_token}"}
    )
    assert response.status_code == 404

def test_update_template_authenticated(test_app_client: TestClient, admin_token: str, mock_pms: MagicMock):
    template_name = "template_to_update"
    update_data = {"description": "Updated description", "template": "New content {var}"}

    # Ensure get_template returns something so it doesn't 404 before update logic
    original_template = ActualPromptTemplate(name=template_name, template="Old", variables=[])
    mock_pms.library.get_template.return_value = original_template

    updated_mock_template = ActualPromptTemplate(name=template_name, template=update_data["template"], description=update_data["description"], variables=["var"], version="1.1") # Assume version updates
    mock_pms.update_template.return_value = updated_mock_template


    response = test_app_client.put(
        f"/api/v1/prompts/templates/{template_name}",
        json=update_data,
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200, response.text
    mock_pms.update_template.assert_called_once_with(
        name=template_name,
        template=update_data["template"],
        description=update_data["description"],
        variables=None # as variables were not in update_data
    )
    assert response.json()["description"] == "Updated description"

def test_delete_template_authenticated(test_app_client: TestClient, admin_token: str, mock_pms: MagicMock):
    template_name = "template_to_delete"
    mock_pms.library.get_template.return_value = MagicMock() # Simulate template exists
    mock_pms.library.delete_template.return_value = True

    response = test_app_client.delete(
        f"/api/v1/prompts/templates/{template_name}",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 204
    mock_pms.library.delete_template.assert_called_once_with(template_name)

# Tests for merged endpoints from prompt_management_router
def test_test_template_endpoint(test_app_client: TestClient, regular_user_token: str, mock_pms: MagicMock):
    template_name = "testable_template"
    variables = {"name": "Jules"}
    mock_template_obj = ActualPromptTemplate(name=template_name, template="Hello {name}!", variables=["name"])
    mock_pms.library.get_template.return_value = mock_template_obj

    response = test_app_client.post(
        "/api/v1/prompts/test",
        json={"template_name": template_name, "variables": variables},
        headers={"Authorization": f"Bearer {regular_user_token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["formatted_prompt"] == "Hello Jules!"
    assert data["template_name"] == template_name

def test_list_template_versions_endpoint(test_app_client: TestClient, regular_user_token: str, mock_pms: MagicMock):
    template_name = "versioned_template"
    # Mock the behavior for get_template_versions, assuming it's on pms.manager
    mock_pms.manager.get_template_versions.return_value = [
        MagicMock(to_dict=MagicMock(return_value={"version": "1.0", "template": "v1", "created_at": "t1", "metrics": {}})),
        MagicMock(to_dict=MagicMock(return_value={"version": "1.1", "template": "v2", "created_at": "t2", "metrics": {}})),
    ]
    # Ensure the base template exists for the initial check in the endpoint
    mock_pms.library.get_template.return_value = MagicMock(name=template_name, version="1.1")


    response = test_app_client.get(
        f"/api/v1/prompts/versions/{template_name}",
        headers={"Authorization": f"Bearer {regular_user_token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["version"] == "1.0"
    assert data[1]["version"] == "1.1"
    # pms.manager.get_template_versions.assert_called_once_with(template_name) # This assertion needs pms.manager to be the direct mock

def test_submit_template_feedback_endpoint(test_app_client: TestClient, regular_user_token: str, mock_pms: MagicMock, test_regular_user: AuthUser):
    template_name = "feedback_template"
    version = "1.0"
    feedback_data = {"rating": 5, "comment": "Great!"}

    # Ensure the base template exists
    mock_pms.library.get_template.return_value = MagicMock(name=template_name)
    mock_pms.manager.record_feedback.return_value = True

    response = test_app_client.post(
        "/api/v1/prompts/feedback",
        json={
            "template_name": template_name,
            "version": version,
            "feedback": feedback_data,
            # user_id can be omitted to use authenticated user's ID
        },
        headers={"Authorization": f"Bearer {regular_user_token}"}
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Feedback recorded successfully"
    mock_pms.manager.record_feedback.assert_called_once_with(
        template_name, version, feedback_data, test_regular_user.id
    )

# TODO: Add tests for /optimize, /usage, /performance endpoints from prompt_router.py
# These will require more setup for mock_pms.get_usage_stats, get_performance_metrics, optimize_prompt methods.
# Example for /usage:
def test_get_usage_statistics_endpoint(test_app_client: TestClient, admin_token: str, mock_pms: MagicMock):
    mock_pms.get_usage_stats.return_value = {"total_usage": 10, "templates": [{"name": "t1", "usage_count": 10, "percentage_of_total": 100.0}]}
    response = test_app_client.get("/api/v1/prompts/usage", headers={"Authorization": f"Bearer {admin_token}"})
    assert response.status_code == 200
    assert response.json()["total_usage"] == 10
    mock_pms.get_usage_stats.assert_called_once_with(None) # No specific template name
