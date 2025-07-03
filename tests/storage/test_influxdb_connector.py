import pytest
from unittest.mock import patch, MagicMock, ANY
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client.client.exceptions import InfluxDBError

# Target for patching get_config_value will depend on where InfluxDBConnector is resolved from.
# Assuming it's directly under Scripts.storage
CONFIG_LOADER_TARGET = "Scripts.storage.influxdb_connector.get_config_value"
INFLUXDB_CLIENT_TARGET = "Scripts.storage.influxdb_connector.InfluxDBClient"

# This import must come AFTER setting up mocks if get_config_value is called at import time by the module
# For now, let's assume it's safe to import here and we patch before instantiation.
from Scripts.storage.influxdb_connector import InfluxDBConnector

@pytest.fixture
def mock_config_values(mocker):
    """Mocks the get_config_value function."""
    config_map = {
        ("INFLUXDB_URL", "time_series_db.influxdb.url", "http://localhost:8086"): "http://mock-influxdb:8086",
        ("INFLUXDB_TOKEN", "time_series_db.influxdb.token", None): "mock_token",
        ("INFLUXDB_ORG", "time_series_db.influxdb.org", None): "mock_org",
        ("INFLUXDB_BUCKET", "time_series_db.influxdb.default_bucket", "my-bucket"): "mock_bucket",
    }
    # Default mock for any other calls
    return mocker.patch(CONFIG_LOADER_TARGET, side_effect=lambda key, yaml_path, default: config_map.get((key, yaml_path, default), default))

@pytest.fixture
def mock_influxdb_client_instance():
    """Mocks the InfluxDBClient instance and its methods."""
    mock_client = MagicMock(spec=InfluxDBClient)
    mock_client.ping.return_value = True # Simulate successful ping

    mock_write_api = MagicMock()
    mock_client.write_api.return_value = mock_write_api

    mock_query_api = MagicMock()
    mock_client.query_api.return_value = mock_query_api

    return mock_client

@pytest.fixture
def influx_connector(mock_config_values, mock_influxdb_client_instance, mocker):
    """Provides an InfluxDBConnector instance with mocked dependencies."""
    # Ensure that InfluxDBConnector._instance is reset for each test
    # to get a fresh instance with new mocks
    if hasattr(InfluxDBConnector, '_instance'):
        InfluxDBConnector._instance = None

    mocker.patch(INFLUXDB_CLIENT_TARGET, return_value=mock_influxdb_client_instance)
    connector = InfluxDBConnector()
    # connector.client = mock_influxdb_client_instance # Ensure the instance uses the mock
    # connector.write_api = mock_influxdb_client_instance.write_api()
    # connector.query_api = mock_influxdb_client_instance.query_api()
    # connector._initialized = True # Manually set as initialized with mocks
    return connector


def test_influxdb_connector_initialization(influx_connector, mock_influxdb_client_instance, mock_config_values):
    """Test successful initialization of InfluxDBConnector."""
    assert influx_connector is not None
    assert influx_connector._initialized
    assert influx_connector.client == mock_influxdb_client_instance
    mock_config_values.assert_any_call("INFLUXDB_URL", "time_series_db.influxdb.url", "http://localhost:8086")
    mock_config_values.assert_any_call("INFLUXDB_TOKEN", "time_series_db.influxdb.token", None)
    mock_config_values.assert_any_call("INFLUXDB_ORG", "time_series_db.influxdb.org", None)
    mock_influxdb_client_instance.ping.assert_called_once()
    mock_influxdb_client_instance.write_api.assert_called_once_with(write_options=SYNCHRONOUS)
    mock_influxdb_client_instance.query_api.assert_called_once()

def test_influxdb_connector_initialization_ping_fails(mock_config_values, mocker):
    """Test initialization when InfluxDB ping fails."""
    if hasattr(InfluxDBConnector, '_instance'): InfluxDBConnector._instance = None
    mock_client_bad_ping = MagicMock(spec=InfluxDBClient)
    mock_client_bad_ping.ping.return_value = False
    mocker.patch(INFLUXDB_CLIENT_TARGET, return_value=mock_client_bad_ping)

    connector = InfluxDBConnector()
    assert not connector._initialized
    assert connector.client is None # Should be None as ping failed
    mock_client_bad_ping.ping.assert_called_once()


def test_write_single_point(influx_connector):
    """Test writing a single point."""
    measurement = "test_measurement"
    tags = {"tag1": "value1"}
    fields = {"field1": 1.0}

    success = influx_connector.write_point(measurement, tags, fields)

    assert success
    influx_connector.write_api.write.assert_called_once()
    # Check the arguments passed to write_api.write
    args, kwargs = influx_connector.write_api.write.call_args
    assert kwargs['bucket'] == "mock_bucket"
    assert kwargs['org'] == "mock_org"
    # The record is a Point object, so checking its structure is more involved.
    # For simplicity, we'll check that it's a Point instance.
    assert isinstance(kwargs['record'], Point)
    # More detailed check on point data:
    point_arg = kwargs['record']
    assert point_arg._name == measurement
    assert point_arg._tags == tags
    assert point_arg._fields == fields


def test_write_single_point_with_timestamp_and_bucket(influx_connector):
    """Test writing a single point with specific timestamp and bucket."""
    measurement = "another_measurement"
    tags = {"host": "hostA"}
    fields = {"value": 123}
    timestamp = 1678886400000000000  # Example nanosecond timestamp
    custom_bucket = "custom_bucket"

    success = influx_connector.write_point(measurement, tags, fields, timestamp=timestamp, bucket=custom_bucket, precision=WritePrecision.NS)

    assert success
    influx_connector.write_api.write.assert_called_once()
    args, kwargs = influx_connector.write_api.write.call_args
    assert kwargs['bucket'] == custom_bucket
    point_arg: Point = kwargs['record']
    assert point_arg._name == measurement
    assert point_arg._tags == tags
    assert point_arg._fields == fields
    assert point_arg._time == timestamp
    assert point_arg._write_precision == WritePrecision.NS


def test_write_points_batch(influx_connector):
    """Test writing a batch of points."""
    point1 = Point("batch_m").tag("t", "v1").field("f", 1)
    point2 = Point("batch_m").tag("t", "v2").field("f", 2)
    records = [point1, point2]

    success = influx_connector.write_points(records)

    assert success
    influx_connector.write_api.write.assert_called_once_with(
        bucket="mock_bucket", org="mock_org", record=records
    )

def test_write_point_api_not_initialized(mock_config_values, mocker):
    """Test write_point when write_api is not initialized (e.g., connection failed initially)."""
    if hasattr(InfluxDBConnector, '_instance'): InfluxDBConnector._instance = None
    mock_client_no_init = MagicMock(spec=InfluxDBClient)
    mock_client_no_init.ping.return_value = False # Simulate initial connection failure
    mocker.patch(INFLUXDB_CLIENT_TARGET, return_value=mock_client_no_init)

    connector = InfluxDBConnector() # This will fail to fully initialize write_api
    assert not connector._initialized
    assert connector.write_api is None

    # Now, mock ping to succeed for the reconnect attempt within write_point
    mock_client_no_init.ping.return_value = True
    mock_write_api = MagicMock()
    mock_client_no_init.write_api.return_value = mock_write_api # Setup write_api for reconnect
    mock_client_no_init.query_api.return_value = MagicMock() # Setup query_api for reconnect

    success = connector.write_point("m", {}, {"f": 1})
    assert success # Should reconnect and succeed
    mock_write_api.write.assert_called_once()


def test_query_data(influx_connector):
    """Test querying data."""
    flux_query = 'from(bucket:"mock_bucket") |> range(start: -1h)'

    # Mock the return value of query_api.query()
    # It should return a list of FluxTable objects, each having a list of FluxRecord objects.
    mock_record1 = MagicMock()
    mock_record1.values = {"_time": "2023-03-15T10:00:00Z", "_value": 10, "host": "serverA"}
    mock_record2 = MagicMock()
    mock_record2.values = {"_time": "2023-03-15T10:01:00Z", "_value": 12, "host": "serverB"}

    mock_table = MagicMock()
    mock_table.records = [mock_record1, mock_record2]

    influx_connector.query_api.query.return_value = [mock_table]

    results = influx_connector.query_data(flux_query)

    assert results is not None
    assert len(results) == 2
    assert results[0] == {"_time": "2023-03-15T10:00:00Z", "_value": 10, "host": "serverA"}
    assert results[1] == {"_time": "2023-03-15T10:01:00Z", "_value": 12, "host": "serverB"}
    influx_connector.query_api.query.assert_called_once_with(query=flux_query, org="mock_org")


def test_query_data_api_error(influx_connector):
    """Test query_data when InfluxDBError occurs."""
    flux_query = 'from(bucket:"mock_bucket") |> range(start: -1h)'
    influx_connector.query_api.query.side_effect = InfluxDBError(response=MagicMock(status=500, reason="Server Error", data="error details"))

    results = influx_connector.query_data(flux_query)

    assert results is None
    influx_connector.query_api.query.assert_called_once_with(query=flux_query, org="mock_org")

def test_query_data_api_not_initialized(mock_config_values, mocker):
    """Test query_data when query_api is not initialized."""
    if hasattr(InfluxDBConnector, '_instance'): InfluxDBConnector._instance = None
    mock_client_no_init = MagicMock(spec=InfluxDBClient)
    mock_client_no_init.ping.return_value = False # Simulate initial connection failure
    mocker.patch(INFLUXDB_CLIENT_TARGET, return_value=mock_client_no_init)

    connector = InfluxDBConnector()
    assert not connector._initialized
    assert connector.query_api is None

    # Mock ping to succeed for the reconnect attempt
    mock_client_no_init.ping.return_value = True
    mock_query_api = MagicMock()
    mock_client_no_init.query_api.return_value = mock_query_api # Setup query_api for reconnect
    mock_client_no_init.write_api.return_value = MagicMock() # Setup write_api for reconnect
    mock_query_api.query.return_value = [] # Simulate query returning empty results

    results = connector.query_data("test query")
    assert results == [] # Should reconnect and succeed (returning empty list from mock)
    mock_query_api.query.assert_called_once()


def test_close_connection(influx_connector):
    """Test closing the connection."""
    influx_connector.close_connection()
    influx_connector.client.close.assert_called_once()
    assert influx_connector.client is None
    assert influx_connector.write_api is None
    assert influx_connector.query_api is None
    assert not influx_connector._initialized
