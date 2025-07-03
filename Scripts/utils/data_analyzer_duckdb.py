import duckdb
import pandas as pd
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DuckDBAnalyzer:
    def __init__(self, db_path: Optional[str] = ':memory:'):
        """
        Initializes the DuckDBAnalyzer.
        Args:
            db_path (Optional[str]): Path to the DuckDB database file.
                                     Defaults to ':memory:' for an in-memory database.
        """
        try:
            self.con = duckdb.connect(database=db_path, read_only=False)
            logger.info(f"DuckDB connection established to '{db_path}'.")
        except Exception as e:
            logger.error(f"Failed to connect to DuckDB at '{db_path}': {e}", exc_info=True)
            raise

    def load_data_from_list_of_dicts(self, data: List[Dict[str, Any]], table_name: str = "source_data"):
        """
        Loads data from a list of dictionaries into a DuckDB table using Pandas as an intermediary.
        Args:
            data (List[Dict[str, Any]]): The data to load.
            table_name (str): The name of the table to create/replace in DuckDB.
        """
        if not data:
            logger.warning(f"No data provided to load into table '{table_name}'.")
            return

        try:
            # Convert list of dicts to Pandas DataFrame
            df = pd.DataFrame(data)
            # Register DataFrame as a DuckDB table. This replaces the table if it exists.
            self.con.register(table_name, df)
            logger.info(f"Successfully loaded {len(df)} rows into DuckDB table '{table_name}'.")
        except Exception as e:
            logger.error(f"Error loading data into DuckDB table '{table_name}': {e}", exc_info=True)
            raise

    def execute_query(self, query: str, params: Optional[List[Any]] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Executes a SQL query on the DuckDB instance.
        Args:
            query (str): The SQL query to execute.
            params (Optional[List[Any]]): Parameters to pass to the query.
        Returns:
            Optional[List[Dict[str, Any]]]: A list of dictionaries representing the query results,
                                             or None if an error occurs.
        """
        try:
            result = self.con.execute(query, parameters=params).fetchall()
            # Convert list of tuples to list of dicts for easier use
            if result:
                column_names = [desc[0] for desc in self.con.description]
                return [dict(zip(column_names, row)) for row in result]
            return []
        except Exception as e:
            logger.error(f"Error executing DuckDB query: {query} with params {params}. Error: {e}", exc_info=True)
            return None

    def analyze_event_data(self, event_data: List[Dict[str, Any]], table_name: str = "events") -> Optional[Dict[str, Any]]:
        """
        Performs a sample analysis on event data: counts events by type and calculates average value.
        Args:
            event_data (List[Dict[str, Any]]): A list of event dictionaries.
                                               Each dict should have 'event_type' (str) and 'value' (numeric).
            table_name (str): Name for the temporary table in DuckDB.
        Returns:
            Optional[Dict[str, Any]]: A dictionary containing analysis results, or None on error.
        """
        if not event_data:
            logger.warning("No event data provided for analysis.")
            return None

        self.load_data_from_list_of_dicts(event_data, table_name)

        # Query 1: Count events by type
        count_query = f"SELECT event_type, COUNT(*) AS event_count FROM {table_name} GROUP BY event_type ORDER BY event_count DESC"
        counts_by_type = self.execute_query(count_query)

        # Query 2: Calculate average value per event type
        avg_value_query = f"SELECT event_type, AVG(value) AS average_value FROM {table_name} GROUP BY event_type"
        avg_values = self.execute_query(avg_value_query)

        # Query 3: Overall average value
        overall_avg_query = f"SELECT AVG(value) AS overall_average_value FROM {table_name}"
        overall_avg_result = self.execute_query(overall_avg_query)
        overall_average_value = overall_avg_result[0]['overall_average_value'] if overall_avg_result and overall_avg_result[0] else None

        if counts_by_type is None or avg_values is None or overall_average_value is None:
            logger.error("Failed to perform one or more analysis queries.")
            return None

        return {
            "counts_by_type": counts_by_type,
            "average_values_by_type": avg_values,
            "overall_average_value": overall_average_value,
            "total_events_analyzed": len(event_data)
        }

    def close(self):
        """Closes the DuckDB connection."""
        if self.con:
            try:
                self.con.close()
                logger.info("DuckDB connection closed.")
            except Exception as e:
                logger.error(f"Error closing DuckDB connection: {e}", exc_info=True)
        self.con = None


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    sample_data = [
        {"event_type": "click", "user_id": "user1", "timestamp": "2023-01-01T10:00:00Z", "value": 10.5, "page": "/home"},
        {"event_type": "view", "user_id": "user2", "timestamp": "2023-01-01T10:01:00Z", "value": 1.0, "page": "/products"},
        {"event_type": "click", "user_id": "user1", "timestamp": "2023-01-01T10:02:00Z", "value": 12.0, "page": "/products/item1"},
        {"event_type": "purchase", "user_id": "user3", "timestamp": "2023-01-01T10:03:00Z", "value": 75.0, "page": "/checkout"},
        {"event_type": "view", "user_id": "user1", "timestamp": "2023-01-01T10:04:00Z", "value": 1.5, "page": "/home"},
        {"event_type": "click", "user_id": "user2", "timestamp": "2023-01-01T10:05:00Z", "value": 9.0, "page": "/contact"},
        {"event_type": "purchase", "user_id": "user1", "timestamp": "2023-01-01T10:06:00Z", "value": 150.2, "page": "/checkout"},
    ]

    analyzer = None
    try:
        # Using in-memory DuckDB for this example
        analyzer = DuckDBAnalyzer()

        logger.info("\n--- Performing analysis on sample event data ---")
        analysis_results = analyzer.analyze_event_data(sample_data)

        if analysis_results:
            logger.info("Analysis Results:")
            logger.info(f"  Total events analyzed: {analysis_results['total_events_analyzed']}")
            logger.info("  Counts by event_type:")
            for item in analysis_results['counts_by_type']:
                logger.info(f"    - {item['event_type']}: {item['event_count']}")
            logger.info("  Average values by event_type:")
            for item in analysis_results['average_values_by_type']:
                logger.info(f"    - {item['event_type']}: {item['average_value']:.2f}")
            logger.info(f"  Overall average value: {analysis_results['overall_average_value']:.2f}")

        logger.info("\n--- Example: Ad-hoc query ---")
        custom_query = "SELECT user_id, SUM(value) AS total_value FROM events WHERE event_type = 'purchase' GROUP BY user_id ORDER BY total_value DESC"
        purchase_totals = analyzer.execute_query(custom_query)
        if purchase_totals:
            logger.info("Total purchase value by user:")
            for row in purchase_totals:
                logger.info(f"  User ID: {row['user_id']}, Total Purchase Value: {row['total_value']:.2f}")

    except Exception as e:
        logger.error(f"An error occurred during DuckDBAnalyzer example: {e}", exc_info=True)
    finally:
        if analyzer:
            analyzer.close()
            logger.info("DuckDBAnalyzer example finished.")
