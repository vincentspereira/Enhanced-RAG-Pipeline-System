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


@pytest.mark.asyncio
async def test_update_admin_config_endpoint(enhanced_api_client, mock_config_manager, admin_token):
    # Ensure ConfigManager is on app.state for the TestClient's app instance
    enhanced_api_client.app.state.config_manager = mock_config_manager

    update_payload = {"api": {"host": "0.0.0.0", "port": 8001}} # Example partial update

    # Mock the config attribute to simulate current state
    current_app_config = AppSystemConfig(
        model=MagicMock(), vector_store=MagicMock(), processing=MagicMock(),
        api=AppAPIConfig(host="127.0.0.1", port=8000, workers=1, max_concurrent_requests=100, default_page_size=10, max_page_size=100), # Provide actual AppAPIConfig
        paths=MagicMock(), cache_settings=MagicMock(), feature_flags=MagicMock(), auth=MagicMock(), elasticsearch=MagicMock()
    )
    mock_config_manager.config = current_app_config
    mock_config_manager.save_config = MagicMock() # Mock save_config

    response = enhanced_api_client.post(
        "/admin/config/update",
        json=update_payload,
        headers={"Authorization": f"Bearer {admin_token}"}
    )

    assert response.status_code == 200
    updated_config_response = response.json()
    assert updated_config_response["api"]["port"] == 8001

    # Verify ConfigManager methods were called
    # Check that the attribute on the existing config object was updated
    assert current_app_config.api.port == 8001 # Check if the actual object was modified
    mock_config_manager.save_config.assert_called_once()


@pytest.mark.asyncio
async def test_update_admin_config_endpoint_unauthorized(enhanced_api_client, user_token):
    response = enhanced_api_client.post(
        "/admin/config/update",
        json={"api": {"port": 8001}},
        headers={"Authorization": f"Bearer {user_token}"} # Regular user token
    )
    assert response.status_code == 403 # Forbidden due to lack of ADMIN_WRITE permission

# --- Tests for RAG Core Endpoints using TestClient & Auth ---

def test_search_rag_endpoint_client_success(enhanced_api_client, admin_token):
    # Access the mocked RAG pipeline from the app state set up by enhanced_api_client fixture
    mock_rag_pipeline_on_client = enhanced_api_client.app.state.rag_pipeline
    mock_rag_pipeline_on_client.search = AsyncMock(return_value=[
        {"text": "client res1", "metadata": {"src": "doc1"}, "score": 0.95}
    ])

    payload = {"query": "client test query", "limit": 1}
    response = enhanced_api_client.post(
        "/search_rag",
        json=payload,
        headers={"Authorization": f"Bearer {admin_token}"} # Assuming SEARCH_RAG needs some auth
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["text"] == "client res1"
    mock_rag_pipeline_on_client.search.assert_awaited_once_with(query="client test query", limit=1, filters=None, categories=None)

def test_search_rag_endpoint_client_unauthorized(enhanced_api_client):
    payload = {"query": "client test query", "limit": 1}
    response = enhanced_api_client.post("/search_rag", json=payload) # No token
    assert response.status_code == 401 # Expecting Unauthorized

def test_process_documents_endpoint_client(enhanced_api_client, admin_token):
    mock_rag_pipeline_on_client = enhanced_api_client.app.state.rag_pipeline
    mock_rag_pipeline_on_client.process_documents = AsyncMock() # Ensure it's an AsyncMock

    payload = {"directory_path": "dummy/client/path", "batch_size": 8}

    # Mock Path().exists() and is_dir() for the client-side call
    with patch('Scripts.enhanced_api.Path') as mock_path_class:
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.is_dir.return_value = True
        mock_path_instance.__str__.return_value = "dummy/client/path" # For the message
        mock_path_class.return_value = mock_path_instance

        response = enhanced_api_client.post(
            "/process_docs",
            json=payload,
            headers={"Authorization": f"Bearer {admin_token}"} # Assuming PROCESS_DOCS needs auth
        )

    assert response.status_code == 200
    assert response.json() == {"message": "Started processing documents from dummy/client/path"}

    # Background tasks are harder to assert directly with TestClient without more complex setup.
    # We trust FastAPI handles it if the endpoint returns success.
    # For unit tests, you'd check `add_task` was called if you directly called the endpoint function.
    # Here, we mainly check the API contract and that the RAG pipeline method would be called.
    # To assert add_task, you'd need to mock BackgroundTasks when the app is created or pass it.
    # For now, let's assume if process_documents is prepared (AsyncMock), it's sufficient for API unit test.

def test_get_system_status_rag_endpoint_client(enhanced_api_client, admin_token):
    mock_rag_pipeline_on_client = enhanced_api_client.app.state.rag_pipeline
    # get_collection_info is sync, but called via run_in_executor in the endpoint
    # The mock on the RAGPipeline instance should just return the value.
    mock_rag_pipeline_on_client.get_collection_info = MagicMock(return_value={"name": "client_coll", "points_count": 200})

    response = enhanced_api_client.get(
        "/system_status_rag",
        headers={"Authorization": f"Bearer {admin_token}"} # Assuming STATUS_RAG needs auth
    )
    assert response.status_code == 200
    assert response.json() == {"name": "client_coll", "points_count": 200}
    mock_rag_pipeline_on_client.get_collection_info.assert_called_once()


def test_generate_response_endpoint_client(enhanced_api_client, admin_token):
    mock_rag_pipeline_on_client = enhanced_api_client.app.state.rag_pipeline
    mock_rag_pipeline_on_client.search = AsyncMock(return_value=[{"text": "context from client", "metadata":{}, "score":0.85}])
    # generate_response is sync in RAGPipeline, called via run_in_executor
    mock_rag_pipeline_on_client.generate_response = MagicMock(return_value={"answer": "RAG client answer", "sources": []})

    payload = {"question": "what is client rag?", "limit": 2}
    response = enhanced_api_client.post(
        "/generate_response",
        json=payload,
        headers={"Authorization": f"Bearer {admin_token}"} # Assuming GENERATE_RAG needs auth
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "RAG client answer"

    mock_rag_pipeline_on_client.search.assert_awaited_once_with(payload["question"], limit=payload["limit"])
    mock_rag_pipeline_on_client.generate_response.assert_called_once_with(
        question=payload["question"],
        search_results=[{"text": "context from client", "metadata":{}, "score":0.85}],
        template_name="qa_prompt" # Default value from Pydantic model
    )

# --- Tests for Copilot Endpoints using TestClient & Auth ---

@pytest.fixture
def mock_copilot_dependencies(enhanced_api_client, mock_copilot_agent, mock_integration_manager):
    """Fixture to mock dependencies for Copilot endpoints via TestClient's app."""
    # Mock the get_integration_manager dependency that's directly used in endpoint
    # It's an async context manager, so mock its __aenter__
    async def mock_get_integration_manager_dependency():
        return mock_integration_manager

    enhanced_api_client.app.dependency_overrides[get_integration_manager] = mock_get_integration_manager_dependency

    # Mock the get_copilot_agent_dependency
    async def mock_get_copilot_agent():
        async with mock_copilot_agent as agent: # Simulate the async context manager behavior
            yield agent

    enhanced_api_client.app.dependency_overrides[get_copilot_agent_dependency] = mock_get_copilot_agent

    # Mock verify_copilot_token_dependency to always allow for these tests
    async def mock_verify_copilot_token():
        return "gca_valid_token_for_test" # Return a valid-looking token
    enhanced_api_client.app.dependency_overrides[verify_copilot_token_dependency] = mock_verify_copilot_token

    yield # Test runs here

    # Clean up overrides
    enhanced_api_client.app.dependency_overrides = {}


def test_consolidated_chat_with_copilot_client(
    enhanced_api_client,
    mock_copilot_dependencies, # This fixture sets up the dependency overrides
    mock_rag_pipeline, # Used by enhanced_api_client's app.state
    mock_copilot_agent, # Mocked agent
    mock_integration_manager, # Mocked integration manager
    admin_token # For FastAPI app auth
):
    # Ensure RAG pipeline is on app.state (if Copilot falls back to it)
    enhanced_api_client.app.state.rag_pipeline = mock_rag_pipeline
    mock_rag_pipeline.search = AsyncMock(return_value=[{"text":"rag_context", "metadata":{}, "score":0.9}])


    payload = {"query": "copilot client query"}
    response = enhanced_api_client.post(
        "/copilot/chat",
        json=payload,
        headers={"Authorization": f"Bearer {admin_token}"} # Assuming COPILOT_CHAT needs auth
    )

    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "copilot_completed"

    mock_integration_manager.optimize_query.assert_awaited_once_with("copilot client query", None) # Context is None if not provided
    mock_rag_pipeline.search.assert_awaited_once() # Context fetch
    mock_copilot_agent.get_completion.assert_awaited_once()


def test_consolidated_stream_chat_with_copilot_client(
    enhanced_api_client,
    mock_copilot_dependencies,
    mock_rag_pipeline,
    mock_copilot_agent,
    mock_integration_manager,
    admin_token
):
    enhanced_api_client.app.state.rag_pipeline = mock_rag_pipeline
    mock_rag_pipeline.search = AsyncMock(return_value=[{"text":"rag_stream_context", "metadata":{}, "score":0.9}])

    # Setup the async generator mock for stream_completion
    async def mock_stream_gen(*args, **kwargs):
        yield "copilot "
        yield "stream "
        yield "response"
    mock_copilot_agent.stream_completion = AsyncMock(return_value=mock_stream_gen())

    payload = {"query": "copilot client stream query"}
    response = enhanced_api_client.post(
        "/copilot/chat/stream",
        json=payload,
        headers={"Authorization": f"Bearer {admin_token}"} # Assuming COPILOT_STREAM needs auth
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    # Collect streamed content
    # TestClient's response.text will concatenate if not iterated properly for streaming
    # For true stream testing, you might need httpx.AsyncClient directly
    # However, for unit testing the endpoint logic:
    content = response.text
    assert "data: {\"content\": \"copilot \"}\n\n" in content
    assert "data: {\"content\": \"stream \"}\n\n" in content
    assert "data: {\"content\": \"response\"}\n\n" in content
    assert "data: [DONE]\n\n" in content

    mock_integration_manager.optimize_query.assert_awaited_once_with("copilot client stream query", None)
    mock_rag_pipeline.search.assert_awaited_once()
    mock_copilot_agent.stream_completion.assert_awaited_once()


# --- Test for Token Endpoint using TestClient ---
def test_login_for_access_token_client_success(enhanced_api_client, mock_auth_manager):
    # Ensure AuthManager is on app.state
    enhanced_api_client.app.state.auth_manager = mock_auth_manager

    mock_user = AuthUser(id="client_user_id", username="client_test", email="ct@e.com", hashed_password="hp", roles=[Role.USER], permissions={Permission.SEARCH_BASIC})
    mock_auth_manager.authenticate_user = AsyncMock(return_value=mock_user)
    mock_auth_manager.create_access_token = MagicMock(return_value="client_jwt_token")

    form_data = {"username": "client_test", "password": "password"} # Use dict for TestClient form data
    response = enhanced_api_client.post("/token", data=form_data)

    assert response.status_code == 200
    data = response.json()
    assert data == {"access_token": "client_jwt_token", "token_type": "bearer"}
    mock_auth_manager.authenticate_user.assert_awaited_once_with("client_test", "password")
    mock_auth_manager.create_access_token.assert_called_once_with(user=mock_user)

def test_login_for_access_token_client_failure(enhanced_api_client, mock_auth_manager):
    enhanced_api_client.app.state.auth_manager = mock_auth_manager
    mock_auth_manager.authenticate_user = AsyncMock(return_value=None) # Simulate auth failure

    form_data = {"username": "wrong_client", "password": "user"}
    response = enhanced_api_client.post("/token", data=form_data)

    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect username or password"


# --- Test for Admin Config Endpoints using TestClient ---
def test_get_admin_config_endpoint_client(enhanced_api_client, mock_config_manager, admin_token):
    # Ensure ConfigManager is on app.state for the TestClient's app instance
    enhanced_api_client.app.state.config_manager = mock_config_manager

    dummy_app_config = AppSystemConfig(
        model=MagicMock(), vector_store=MagicMock(), processing=MagicMock(), api=MagicMock(),
        paths=MagicMock(), cache_settings=MagicMock(), feature_flags=MagicMock(), auth=MagicMock(), elasticsearch=MagicMock()
    )
    # Pydantic models need to be converted to dict for JSON response comparison
    dummy_config_dict = dummy_app_config.dict()

    # Instead of mocking config_manager.config, we mock the .dict() method of the config object
    # because the endpoint returns config_manager_instance.config which is then serialized by FastAPI
    mock_config_manager.config = MagicMock(spec=AppSystemConfig)
    mock_config_manager.config.dict.return_value = dummy_config_dict # Mock .dict()

    response = enhanced_api_client.get(
        "/admin/config",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200
    assert response.json() == dummy_config_dict

def test_get_admin_config_endpoint_unauthorized(enhanced_api_client, user_token):
    response = enhanced_api_client.get(
        "/admin/config",
        headers={"Authorization": f"Bearer {user_token}"} # Regular user token
    )
    assert response.status_code == 403 # Forbidden


# --- Test for verify_copilot_token_dependency ---
@pytest.mark.asyncio
async def test_verify_copilot_token_dependency_valid():
    token = "gca_validtoken123"
    result = await verify_copilot_token_dependency(x_copilot_token=token)
    assert result == token

@pytest.mark.asyncio
async def test_verify_copilot_token_dependency_none():
    result = await verify_copilot_token_dependency(x_copilot_token=None)
    assert result is None # Should allow None if header is not present

@pytest.mark.asyncio
async def test_verify_copilot_token_dependency_invalid_format():
    with pytest.raises(HTTPException) as excinfo:
        await verify_copilot_token_dependency(x_copilot_token="invalid_token")
    assert excinfo.value.status_code == 401
    assert "Invalid Copilot Agent token format" in excinfo.value.detail


# Placeholder for other system endpoints like /health, /status, /search_direct_qdrant etc.
# These would also use enhanced_api_client and mock dependencies as needed.
# Example:
def test_health_check_endpoint_main(enhanced_api_client, mock_auth_manager):
    # Mock dependencies for health check if any are complex
    # For instance, if check_qdrant_health or check_embedding_provider_health
    # are called, their underlying calls might need mocking or the dependencies
    # themselves might be overridden.
    # For this example, assume basic check.

    # The health endpoint might rely on app.state.integration_manager_instance etc.
    # Ensure these are available or mocked if the endpoint tries to access them.
    mock_integration_mgr = MagicMock()
    mock_integration_mgr.get_integration_stats = MagicMock(return_value={"latest_health": {"cpu_usage": 0.1, "memory_usage": 0.2}})
    enhanced_api_client.app.state.integration_manager_instance = mock_integration_mgr

    mock_recovery_system = MagicMock()
    mock_recovery_system.get_system_health = MagicMock(return_value={"overall_state": "healthy"})

    # Mock get_error_recovery_system if it's used by the health endpoint
    with patch('Scripts.enhanced_api.get_error_recovery_system', return_value=mock_recovery_system), \
         patch('Scripts.enhanced_api.check_qdrant_health', AsyncMock(return_value={"status": "healthy"})), \
         patch('Scripts.enhanced_api.check_embedding_provider_health', AsyncMock(return_value={"status": "healthy"})):

        response = enhanced_api_client.get("/health") # No auth typically for health

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["components"]["qdrant"] == "healthy"

# Final considerations:
# - Test all error paths (e.g., RAG pipeline not initialized, directory not found for process_docs).
# - Test input validation (e.g., invalid payload for search_rag).
# - For endpoints with BackgroundTasks, direct assertion of task calls is tricky with TestClient.
#   Unit testing the endpoint function directly (as in the original tests) can be better for that specific aspect,
#   or using more advanced techniques like spying on `background_tasks.add_task`.
# - The current `enhanced_api_client` fixture mocks RAGPipeline, AuthManager, ConfigManager at the app.state level.
#   This is good for isolating API layer logic. Integration tests would use real instances.
