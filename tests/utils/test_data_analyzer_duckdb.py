import pytest
import duckdb
import pandas as pd

# Assuming data_analyzer_duckdb.py is in Scripts/utils/
from Scripts.utils.data_analyzer_duckdb import DuckDBAnalyzer

@pytest.fixture
def sample_event_data():
    """Provides sample data for testing."""
    return [
        {"event_type": "click", "user_id": "user1", "value": 10.5, "page": "/home"},
        {"event_type": "view", "user_id": "user2", "value": 1.0, "page": "/products"},
        {"event_type": "click", "user_id": "user1", "value": 12.0, "page": "/products/item1"},
        {"event_type": "purchase", "user_id": "user3", "value": 75.0, "page": "/checkout"},
        {"event_type": "view", "user_id": "user1", "value": 1.5, "page": "/home"},
        {"event_type": "click", "user_id": "user2", "value": 9.0, "page": "/contact"},
        {"event_type": "purchase", "user_id": "user1", "value": 150.2, "page": "/checkout"},
    ]

@pytest.fixture
def duckdb_analyzer():
    """Provides a DuckDBAnalyzer instance with an in-memory database."""
    analyzer = DuckDBAnalyzer() # Default is :memory:
    yield analyzer
    analyzer.close() # Ensure connection is closed after test

def test_duckdb_analyzer_initialization():
    """Test successful initialization of DuckDBAnalyzer."""
    analyzer = DuckDBAnalyzer()
    assert analyzer.con is not None
    # Check if it's an in-memory database by default
    res = analyzer.con.execute("PRAGMA database_list;").fetchall()
    assert len(res) == 1
    assert res[0][1] == 'main' # Default main db
    # Path for in-memory is empty string in PRAGMA database_list result for file column
    # For DuckDB versions < 0.7.1, it might be ':memory:'. For >= 0.7.1, it's an empty string.
    assert res[0][2] == '' or res[0][2] == ':memory:'
    analyzer.close()

def test_load_data_from_list_of_dicts(duckdb_analyzer, sample_event_data):
    """Test loading data from a list of dictionaries."""
    table_name = "test_events"
    duckdb_analyzer.load_data_from_list_of_dicts(sample_event_data, table_name)

    # Verify data was loaded
    result = duckdb_analyzer.con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    assert result is not None
    assert result[0] == len(sample_event_data)

    # Verify columns (Pandas infers types, DuckDB adapts)
    df_from_duckdb = duckdb_analyzer.con.table(table_name).df()
    pd_df = pd.DataFrame(sample_event_data)
    pd.testing.assert_frame_equal(df_from_duckdb, pd_df, check_dtype=False) # Dtypes might differ slightly

def test_load_data_empty_list(duckdb_analyzer, caplog):
    """Test loading an empty list of data."""
    duckdb_analyzer.load_data_from_list_of_dicts([], "empty_table")
    assert "No data provided to load into table 'empty_table'" in caplog.text

    # Check that table might not exist or is empty
    with pytest.raises(duckdb.CatalogException): # Or duckdb.IOException depending on version/exact error
        duckdb_analyzer.con.execute("SELECT * FROM empty_table").fetchall()
    # A more robust check might be to see if the table exists in information_schema.tables

def test_execute_query_select(duckdb_analyzer, sample_event_data):
    """Test executing a SELECT query."""
    table_name = "select_test_data"
    duckdb_analyzer.load_data_from_list_of_dicts(sample_event_data, table_name)

    query = f"SELECT user_id, page FROM {table_name} WHERE event_type = 'click'"
    results = duckdb_analyzer.execute_query(query)

    assert results is not None
    assert len(results) == 3
    expected_users = {"user1", "user1", "user2"}
    found_users = {res['user_id'] for res in results}
    # Note: order is not guaranteed by SQL unless ORDER BY is used.
    # So checking counts and content is better.
    assert len([r for r in results if r['user_id'] == 'user1']) == 2
    assert len([r for r in results if r['user_id'] == 'user2']) == 1


def test_execute_query_aggregation(duckdb_analyzer, sample_event_data):
    """Test executing an aggregation query."""
    table_name = "agg_test_data"
    duckdb_analyzer.load_data_from_list_of_dicts(sample_event_data, table_name)

    query = f"SELECT event_type, COUNT(*) as count, AVG(value) as avg_value FROM {table_name} GROUP BY event_type"
    results = duckdb_analyzer.execute_query(query)

    assert results is not None
    assert len(results) == 3 # click, view, purchase

    for res_dict in results:
        if res_dict['event_type'] == 'click':
            assert res_dict['count'] == 3
            assert pytest.approx(res_dict['avg_value']) == (10.5 + 12.0 + 9.0) / 3
        elif res_dict['event_type'] == 'purchase':
            assert res_dict['count'] == 2
            assert pytest.approx(res_dict['avg_value']) == (75.0 + 150.2) / 2

def test_execute_query_with_parameters(duckdb_analyzer, sample_event_data):
    """Test executing a query with parameters."""
    table_name = "param_test_data"
    duckdb_analyzer.load_data_from_list_of_dicts(sample_event_data, table_name)

    query = f"SELECT event_type, value FROM {table_name} WHERE user_id = ? AND value > ?"
    params = ["user1", 11.0]
    results = duckdb_analyzer.execute_query(query, params)

    assert results is not None
    assert len(results) == 2 # (click, 12.0), (purchase, 150.2) for user1 with value > 11.0
    event_types_found = {res['event_type'] for res in results}
    assert "click" in event_types_found
    assert "purchase" in event_types_found


def test_execute_query_invalid_sql(duckdb_analyzer, caplog):
    """Test executing an invalid SQL query."""
    results = duckdb_analyzer.execute_query("SELECT FROM table_that_does_not_exist")
    assert results is None
    assert "Error executing DuckDB query" in caplog.text

def test_analyze_event_data_success(duckdb_analyzer, sample_event_data):
    """Test the analyze_event_data method for successful analysis."""
    analysis = duckdb_analyzer.analyze_event_data(sample_event_data)

    assert analysis is not None
    assert analysis["total_events_analyzed"] == len(sample_event_data)

    # Check counts by type
    counts = {item['event_type']: item['event_count'] for item in analysis["counts_by_type"]}
    assert counts.get("click") == 3
    assert counts.get("view") == 2
    assert counts.get("purchase") == 2

    # Check average values by type
    avg_values = {item['event_type']: item['average_value'] for item in analysis["average_values_by_type"]}
    assert pytest.approx(avg_values.get("click")) == (10.5 + 12.0 + 9.0) / 3
    assert pytest.approx(avg_values.get("purchase")) == (75.0 + 150.2) / 2

    # Check overall average value
    all_values = [d['value'] for d in sample_event_data]
    expected_overall_avg = sum(all_values) / len(all_values)
    assert pytest.approx(analysis["overall_average_value"]) == expected_overall_avg

def test_analyze_event_data_empty_input(duckdb_analyzer, caplog):
    """Test analyze_event_data with empty input."""
    analysis = duckdb_analyzer.analyze_event_data([])
    assert analysis is None
    assert "No event data provided for analysis." in caplog.text

def test_duckdb_analyzer_close_connection(duckdb_analyzer):
    """Test closing the DuckDB connection."""
    assert duckdb_analyzer.con is not None
    duckdb_analyzer.close()
    assert duckdb_analyzer.con is None
    # Try to use connection after close should fail or re-open (depending on impl)
    # Current DuckDBAnalyzer does not auto-reopen on execute_query if con is None
    with pytest.raises(AttributeError): # 'NoneType' object has no attribute 'execute'
         duckdb_analyzer.execute_query("SELECT 1")

# To run these tests:
# Ensure pytest is installed.
# Run from the root directory: pytest tests/utils/test_data_analyzer_duckdb.py
