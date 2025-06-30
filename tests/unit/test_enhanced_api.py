import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import HTTPException, BackgroundTasks, Request as FastAPIRequest, Header
from fastapi.testclient import TestClient # Though for unit tests, we often call handlers directly
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Union

# Modules to test or mock
from Scripts.enhanced_api import (
    app, # The FastAPI app instance
    search_rag_endpoint,
    process_documents_endpoint,
    get_system_status_rag_endpoint,
    generate_response_endpoint,
    consolidated_chat_with_copilot,
    consolidated_stream_chat_with_copilot,
    get_admin_config_endpoint,
    update_admin_config_endpoint,
    login_for_access_token,
    SearchQueryInput, ProcessRequestInput, GenerateRequestInput, MigratedSearchResponse, SearchResultItem,
    ConfigUpdateRequestModel,
    verify_copilot_token_dependency # The dependency itself
)
from Scripts.rag_pipeline import RAGPipeline
from Scripts.auth.auth_manager import AuthManager, AuthUser, Permission, Role
from Scripts.config.manager import ConfigManager, SystemConfig as AppSystemConfig
from Scripts.integrations import CopilotAgent, CopilotRequest, CopilotResponse


# --- Mocks for dependencies ---
@pytest.fixture
def mock_rag_pipeline():
    mock = MagicMock(spec=RAGPipeline)
    mock.search = AsyncMock(return_value=[{"text": "searched", "metadata": {}, "score": 0.9}])
    mock.process_documents = AsyncMock()
    mock.get_collection_info = MagicMock(return_value={"name": "test_coll", "points_count": 10})
    # generate_response is sync in RAGPipeline but called via run_in_executor, so mock its direct return
    mock.generate_response = MagicMock(return_value={"answer": "generated_answer", "sources": []})
    mock.async_initialize_components = AsyncMock()
    return mock

@pytest.fixture
def mock_auth_manager():
    mock = MagicMock(spec=AuthManager)
    mock.authenticate_user = AsyncMock()
    mock.create_access_token = MagicMock(return_value="test_jwt_token")
    # For require_permission, the dependency itself will be mocked in specific tests
    return mock

@pytest.fixture
def mock_config_manager():
    mock = MagicMock(spec=ConfigManager)
    mock.config = MagicMock(spec=AppSystemConfig) # Mock the config attribute
    mock.update_config = MagicMock()
    return mock

@pytest.fixture
def mock_copilot_agent():
    mock = MagicMock(spec=CopilotAgent)
    mock.get_completion = AsyncMock(return_value=CopilotResponse(response="copilot_completed", context_used=[], tokens_used=10, model="copilot-test"))
    mock.stream_completion = AsyncMock() # For stream, need to make it an async generator

    async def mock_stream_gen(*args, **kwargs):
        yield "copilot "
        yield "stream "
        yield "response"
    mock.stream_completion.return_value = mock_stream_gen()
    return mock

@pytest.fixture
def mock_integration_manager():
    mock = AsyncMock()
    mock.optimize_query = AsyncMock(side_effect=lambda query, context: {"optimized_query": query, "metadata": {}}) # Pass through query
    return mock

@pytest.fixture
def mock_fastapi_request(mock_rag_pipeline, mock_auth_manager, mock_config_manager):
    # Create a mock FastAPI Request object with app.state
    mock_req = MagicMock(spec=FastAPIRequest)
    mock_req.app = MagicMock()
    mock_req.app.state = MagicMock()
    mock_req.app.state.rag_pipeline = mock_rag_pipeline
    mock_req.app.state.auth_manager = mock_auth_manager
    mock_req.app.state.config_manager = mock_config_manager
    # Mock other app.state attributes if endpoints directly use them
    mock_req.app.state.app_config = MagicMock(spec=AppSystemConfig) # For endpoints using app_config directly
    mock_req.app.state.app_config.feature_flags = MagicMock(enable_elasticsearch_fallback=True) # Example
    return mock_req


# --- Unit Tests for Endpoints ---

@pytest.mark.asyncio
async def test_search_rag_endpoint_success(mock_fastapi_request, mock_rag_pipeline):
    payload = SearchQueryInput(query="test query", limit=3)
    mock_rag_pipeline.search.return_value = [
        {"text": "res1", "metadata": {"src": "doc1"}, "score": 0.95}
    ]

    response = await search_rag_endpoint(payload, mock_fastapi_request)

    mock_rag_pipeline.search.assert_awaited_once_with(query="test query", limit=3, filters=None, categories=None)
    assert len(response.results) == 1
    assert response.results[0].text == "res1"

@pytest.mark.asyncio
async def test_search_rag_endpoint_pipeline_not_init(mock_fastapi_request):
    mock_fastapi_request.app.state.rag_pipeline = None
    payload = SearchQueryInput(query="test query")
    with pytest.raises(HTTPException) as excinfo:
        await search_rag_endpoint(payload, mock_fastapi_request)
    assert excinfo.value.status_code == 503

@pytest.mark.asyncio
async def test_process_documents_endpoint(mock_fastapi_request, mock_rag_pipeline):
    payload = ProcessRequestInput(directory_path="dummy/path", batch_size=16)
    # BackgroundTasks can be mocked or a real one passed if its methods aren't critical to unit logic
    mock_background_tasks = MagicMock(spec=BackgroundTasks)

    # Mock Path().exists() and is_dir()
    with patch('Scripts.enhanced_api.Path') as mock_path_class:
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.is_dir.return_value = True
        mock_path_class.return_value = mock_path_instance

        response = await process_documents_endpoint(payload, mock_background_tasks, mock_fastapi_request)

    mock_background_tasks.add_task.assert_called_once_with(
        mock_rag_pipeline.process_documents,
        mock_path_instance, # Check that the Path("dummy/path") is passed
        16
    )
    assert response == {"message": "Started processing documents from dummy/path"}


@pytest.mark.asyncio
async def test_get_system_status_rag_endpoint(mock_fastapi_request, mock_rag_pipeline):
    # get_collection_info is sync, so mock its direct return value
    mock_rag_pipeline.get_collection_info.return_value = {"name": "test_coll", "points_count": 120}

    # Mock run_in_executor for the sync call
    with patch('asyncio.get_event_loop') as mock_get_loop:
        mock_loop = MagicMock()
        mock_loop.run_in_executor = AsyncMock(return_value={"name": "test_coll", "points_count": 120}) # Simulating it returns the expected dict
        mock_get_loop.return_value = mock_loop

        response = await get_system_status_rag_endpoint(mock_fastapi_request) # type: ignore

    mock_loop.run_in_executor.assert_awaited_once_with(None, mock_rag_pipeline.get_collection_info)
    assert response == {"name": "test_coll", "points_count": 120}


@pytest.mark.asyncio
async def test_generate_response_endpoint_async(mock_fastapi_request, mock_rag_pipeline): # Renamed test
    payload = GenerateRequestInput(question="what is rag?", limit=3)

    # Mock the search call that the endpoint will make first
    mock_rag_pipeline.search = AsyncMock(return_value=[{"text": "context", "metadata":{}, "score":0.8}])

    # Mock the now-async generate_response method
    mock_rag_pipeline.generate_response = AsyncMock(return_value={"answer": "RAG is async...", "sources": []})

    response = await generate_response_endpoint(payload, mock_fastapi_request) # type: ignore

    mock_rag_pipeline.search.assert_awaited_once_with(payload.question, limit=payload.limit)
    mock_rag_pipeline.generate_response.assert_awaited_once_with(
        question=payload.question,
        search_results=mock_rag_pipeline.search.return_value, # The result of the awaited search
        template_name=payload.template_name
    )
    assert response["answer"] == "RAG is async..."


@pytest.mark.asyncio
async def test_consolidated_chat_with_copilot(mock_fastapi_request, mock_rag_pipeline, mock_copilot_agent):
    copilot_req_body = CopilotRequest(query="copilot query")

    with patch('Scripts.enhanced_api.get_integration_manager', AsyncMock(return_value=mock_integration_manager())) as pim_mock, \
         patch('Scripts.enhanced_api.get_copilot_agent_dependency', return_value=mock_copilot_agent): # Mock the dependency directly

        response = await consolidated_chat_with_copilot(
            copilot_req_body,
            agent=mock_copilot_agent, # Pass the mocked agent
            copilot_token=None, # Assuming token is optional or valid for this test
            fastapi_req=mock_fastapi_request
        )

    pim_mock.return_value.optimize_query.assert_awaited_once_with("copilot query", [])
    mock_rag_pipeline.search.assert_awaited_once() # Context fetch
    mock_copilot_agent.get_completion.assert_awaited_once()
    assert response.response == "copilot_completed"


@pytest.mark.asyncio
async def test_login_for_access_token_success(mock_fastapi_request, mock_auth_manager):
    mock_user = AuthUser(id="1", username="test", email="t@e.com", hashed_password="hp", roles=[Role.USER], permissions={Permission.SEARCH_BASIC})
    mock_auth_manager.authenticate_user = AsyncMock(return_value=mock_user)
    form_data = MagicMock(spec=OAuth2PasswordRequestForm)
    form_data.username = "test"
    form_data.password = "pass"

    response = await login_for_access_token(mock_fastapi_request, form_data, mock_auth_manager)

    mock_auth_manager.authenticate_user.assert_awaited_once_with("test", "pass")
    mock_auth_manager.create_access_token.assert_called_once_with(user=mock_user)
    assert response == {"access_token": "test_jwt_token", "token_type": "bearer"}

@pytest.mark.asyncio
async def test_login_for_access_token_failure(mock_fastapi_request, mock_auth_manager):
    mock_auth_manager.authenticate_user = AsyncMock(return_value=None) # Simulate auth failure
    form_data = MagicMock(spec=OAuth2PasswordRequestForm)
    form_data.username = "wrong"
    form_data.password = "user"

    with pytest.raises(HTTPException) as excinfo:
        await login_for_access_token(mock_fastapi_request, form_data, mock_auth_manager)
    assert excinfo.value.status_code == 401 # fastapi_status.HTTP_401_UNAUTHORIZED

@pytest.mark.asyncio
async def test_get_admin_config_endpoint(mock_fastapi_request, mock_config_manager, mock_auth_manager):
    # Mock the require_permission dependency to simulate an authorized admin user
    mock_admin_user = AuthUser(id="admin_id", username="admin", email="admin@e.com", hashed_password="hp", roles=[Role.ADMIN], permissions=set(Permission))

    # This is tricky. The dependency is `Depends(get_auth_manager).require_permission(...)`.
    # We need to mock what `require_permission` returns when called.
    # The `get_auth_manager` fixture would return `mock_auth_manager`.
    # So, `mock_auth_manager.require_permission(Permission.ADMIN_READ)` should return a callable,
    # and that callable, when Depends resolves it, should return `mock_admin_user`.

    mock_permission_dependency = MagicMock(return_value=mock_admin_user)
    mock_auth_manager.require_permission.return_value = mock_permission_dependency
    mock_fastapi_request.app.state.auth_manager = mock_auth_manager # Ensure it's on app.state

    # Setup a dummy config on the mocked config_manager
    dummy_app_config = AppSystemConfig(
        model=MagicMock(), vector_store=MagicMock(), processing=MagicMock(), api=MagicMock(),
        paths=MagicMock(), cache_settings=MagicMock(), feature_flags=MagicMock(), auth=MagicMock(), elasticsearch=MagicMock()
    )
    mock_config_manager.config = dummy_app_config
    mock_fastapi_request.app.state.config_manager = mock_config_manager

    # The Depends for current_user will use the mocked setup
    response = await get_admin_config_endpoint(mock_fastapi_request, current_user=mock_admin_user) # Pass current_user directly

    mock_auth_manager.require_permission.assert_called_with(Permission.ADMIN_READ)
    # The dependency itself is called by FastAPI, so mock_permission_dependency should have been called
    # This part of testing Depends is often better with TestClient.
    assert response == dummy_app_config

# More tests would be needed for update_admin_config, other Copilot cases, error conditions, etc.
# This provides a starting structure.
