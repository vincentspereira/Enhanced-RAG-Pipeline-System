import pytest
from unittest.mock import patch, MagicMock, ANY
from neo4j import GraphDatabase, Driver, Result, SummaryCounters
from neo4j.exceptions import Neo4jError, ServiceUnavailable

# Target for patching get_config_value will depend on where Neo4jConnector is resolved from.
CONFIG_LOADER_TARGET = "Scripts.storage.neo4j_connector.get_config_value"
NEO4J_DRIVER_TARGET = "Scripts.storage.neo4j_connector.GraphDatabase.driver"

from Scripts.storage.neo4j_connector import Neo4jConnector

@pytest.fixture
def mock_config_values_neo4j(mocker):
    """Mocks the get_config_value function for Neo4j."""
    config_map = {
        ("NEO4J_URI", "graph_db.neo4j.uri", "bolt://localhost:7687"): "bolt://mock-neo4j:7687",
        ("NEO4J_USER", "graph_db.neo4j.user", "neo4j"): "mock_user",
        ("NEO4J_PASSWORD", "graph_db.neo4j.password", None): "mock_password", # Ensure this is always returned
    }
    return mocker.patch(CONFIG_LOADER_TARGET, side_effect=lambda key, yaml_path, default: config_map.get((key, yaml_path, default), default))

@pytest.fixture
def mock_neo4j_driver_instance():
    """Mocks the Neo4j Driver instance and its session methods."""
    mock_driver = MagicMock(spec=Driver)
    mock_session = MagicMock()
    mock_driver.session.return_value = mock_session # session() is a context manager
    mock_session.__enter__.return_value = mock_session # For 'with driver.session() as session:'
    mock_session.__exit__.return_value = None

    # Mock read_transaction and write_transaction to return a mock Result
    mock_result = MagicMock(spec=Result)
    mock_result.consume.return_value.summary = MagicMock(spec=SummaryCounters) # For write summary

    # Make mock_result iterable (for read_transaction)
    mock_result.__iter__.return_value = iter([]) # Default to no records

    mock_session.read_transaction.return_value = mock_result
    mock_session.write_transaction.return_value = mock_result

    # Mock the initial verification query in __init__
    mock_init_result = MagicMock(spec=Result)
    mock_init_result.consume.return_value = MagicMock() # consume() returns a summary-like object
    mock_session.run.return_value = mock_init_result # For the "RETURN 1" check

    return mock_driver

@pytest.fixture
def neo4j_connector(mock_config_values_neo4j, mock_neo4j_driver_instance, mocker):
    """Provides a Neo4jConnector instance with mocked dependencies."""
    if hasattr(Neo4jConnector, '_instance'):
        Neo4jConnector._instance = None # Reset singleton for fresh instance

    mocker.patch(NEO4J_DRIVER_TARGET, return_value=mock_neo4j_driver_instance)
    connector = Neo4jConnector()
    return connector

def test_neo4j_connector_initialization(neo4j_connector, mock_neo4j_driver_instance, mock_config_values_neo4j):
    """Test successful initialization of Neo4jConnector."""
    assert neo4j_connector is not None
    assert neo4j_connector._initialized
    assert neo4j_connector.driver == mock_neo4j_driver_instance
    mock_config_values_neo4j.assert_any_call("NEO4J_URI", "graph_db.neo4j.uri", "bolt://localhost:7687")
    mock_config_values_neo4j.assert_any_call("NEO4J_USER", "graph_db.neo4j.user", "neo4j")
    mock_config_values_neo4j.assert_any_call("NEO4J_PASSWORD", "graph_db.neo4j.password", None)

    # Check that GraphDatabase.driver was called with correct auth
    mock_neo4j_driver_instance.session.assert_called() # Used for initial connection verification


def test_neo4j_connector_initialization_service_unavailable(mock_config_values_neo4j, mocker):
    """Test initialization when Neo4j service is unavailable."""
    if hasattr(Neo4jConnector, '_instance'): Neo4jConnector._instance = None
    mocker.patch(NEO4J_DRIVER_TARGET, side_effect=ServiceUnavailable("Mock Service Unavailable"))

    connector = Neo4jConnector()
    assert not connector._initialized
    assert connector.driver is None

def test_neo4j_connector_initialization_auth_error(mock_config_values_neo4j, mocker):
    """Test initialization with Neo4j authentication error."""
    if hasattr(Neo4jConnector, '_instance'): Neo4jConnector._instance = None
    # Simulate an auth error by raising Neo4jError (actual auth errors might be more specific)
    mocker.patch(NEO4J_DRIVER_TARGET, side_effect=Neo4jError("Mock Authentication Failed"))

    connector = Neo4jConnector()
    assert not connector._initialized
    assert connector.driver is None

def test_execute_read_query(neo4j_connector, mock_neo4j_driver_instance):
    """Test executing a read query."""
    cypher_query = "MATCH (n) RETURN n.name AS name"
    params = {"limit": 10}

    # Setup mock result for read_transaction
    mock_record1 = {"name": "Alice"}
    mock_record2 = {"name": "Bob"}
    mock_result_iterable = MagicMock(spec=Result)
    mock_result_iterable.__iter__.return_value = iter([mock_record1, mock_record2]) # Simulate two records

    mock_session = mock_neo4j_driver_instance.session.return_value.__enter__.return_value
    mock_session.read_transaction.return_value = mock_result_iterable

    results = neo4j_connector.execute_read_query(cypher_query, params)

    assert results is not None
    assert len(results) == 2
    assert results[0] == {"name": "Alice"}
    assert results[1] == {"name": "Bob"}
    # The actual function passed to read_transaction is Neo4jConnector._run_cypher_query
    # We check that read_transaction was called. The first arg to it is the function.
    mock_session.read_transaction.assert_called_once_with(ANY, cypher_query, params)


def test_execute_read_query_error(neo4j_connector, mock_neo4j_driver_instance):
    """Test read query execution when Neo4jError occurs."""
    cypher_query = "MATCH (n) RETURN n.name"
    mock_session = mock_neo4j_driver_instance.session.return_value.__enter__.return_value
    mock_session.read_transaction.side_effect = Neo4jError("Read query failed")

    results = neo4j_connector.execute_read_query(cypher_query)
    assert results is None

def test_execute_write_query(neo4j_connector, mock_neo4j_driver_instance):
    """Test executing a write query."""
    cypher_query = "CREATE (p:Person {name: $name})"
    params = {"name": "Charlie"}

    mock_summary = MagicMock(spec=SummaryCounters)
    mock_summary.counters.nodes_created = 1

    mock_result_summary = MagicMock(spec=Result)
    mock_result_summary.consume.return_value.summary = mock_summary

    mock_session = mock_neo4j_driver_instance.session.return_value.__enter__.return_value
    mock_session.write_transaction.return_value = mock_result_summary

    summary_result = neo4j_connector.execute_write_query(cypher_query, params)

    assert summary_result is not None
    assert summary_result.counters.nodes_created == 1
    mock_session.write_transaction.assert_called_once_with(ANY, cypher_query, params)

def test_execute_write_query_error(neo4j_connector, mock_neo4j_driver_instance):
    """Test write query execution when Neo4jError occurs."""
    cypher_query = "CREATE (p:Person {name: $name})"
    mock_session = mock_neo4j_driver_instance.session.return_value.__enter__.return_value
    mock_session.write_transaction.side_effect = Neo4jError("Write query failed")

    summary = neo4j_connector.execute_write_query(cypher_query, {"name": "David"})
    assert summary is None

def test_create_node_helper(neo4j_connector, mock_neo4j_driver_instance):
    """Test the create_node helper method."""
    label = "User"
    properties = {"username": "eve", "email": "eve@example.com"}

    mock_summary = MagicMock(spec=SummaryCounters)
    mock_summary.counters.nodes_created = 1

    mock_result_summary = MagicMock(spec=Result)
    mock_result_summary.consume.return_value.summary = mock_summary

    mock_session = mock_neo4j_driver_instance.session.return_value.__enter__.return_value
    mock_session.write_transaction.return_value = mock_result_summary

    result_summary = neo4j_connector.create_node(label, properties)
    assert result_summary is not None
    assert result_summary.counters.nodes_created == 1
    # Check that the correct Cypher query was constructed for CREATE
    # The first argument to write_transaction is the function, the second is the query
    args, kwargs = mock_session.write_transaction.call_args
    assert "CREATE (n:User $props)" in args[1]
    assert args[2] == {"props": properties}


def test_create_relationship_helper(neo4j_connector, mock_neo4j_driver_instance):
    """Test the create_relationship helper method."""
    mock_summary = MagicMock(spec=SummaryCounters)
    mock_summary.counters.relationships_created = 1

    mock_result_summary = MagicMock(spec=Result)
    mock_result_summary.consume.return_value.summary = mock_summary

    mock_session = mock_neo4j_driver_instance.session.return_value.__enter__.return_value
    mock_session.write_transaction.return_value = mock_result_summary

    success = neo4j_connector.create_relationship(
        "User", {"name": "eve"},
        "Resource", {"name": "doc1"},
        "OWNS", {"since": "2023"}
    )
    assert success
    args, kwargs = mock_session.write_transaction.call_args
    assert "MATCH (a:User {name: $start_name}), (b:Resource {name: $end_name})" in args[1]
    assert "MERGE (a)-[r:OWNS]->(b)" in args[1]
    assert "SET r += $rel_props" in args[1]
    assert args[2] == {"start_name": "eve", "end_name": "doc1", "rel_props": {"since": "2023"}}


def test_close_connection(neo4j_connector, mock_neo4j_driver_instance):
    """Test closing the connection."""
    assert neo4j_connector.driver is not None # Should be set by fixture
    neo4j_connector.close()
    mock_neo4j_driver_instance.close.assert_called_once()
    assert neo4j_connector.driver is None
    assert not neo4j_connector._initialized
