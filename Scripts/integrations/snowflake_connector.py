import os
import snowflake.connector
from snowflake.connector.errors import ProgrammingError, OperationalError
import logging
from typing import List, Dict, Any, Optional

# Configure logger
logger = logging.getLogger(__name__)

# Import config loader
try:
    # Assuming this script is in Scripts/integrations/ and config_loader is in Scripts/utils/
    from Scripts.utils.config_loader import get_config_value
except ImportError:
    import sys
    # Adjust path to go up one level from 'integrations' to 'Scripts', then into 'utils'
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from utils.config_loader import get_config_value
    except ImportError as e:
        logger.error(f"Critical: Failed to import get_config_value for SnowflakeConnector. Error: {e}")
        def get_config_value(env_var_name, yaml_path=None, default=None): # Basic fallback
            return os.getenv(env_var_name, default)

class SnowflakeConnector:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SnowflakeConnector, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.account = get_config_value("SNOWFLAKE_ACCOUNT", yaml_path="data_warehouse.snowflake.account")
        self.user = get_config_value("SNOWFLAKE_USER", yaml_path="data_warehouse.snowflake.user")
        self.password = get_config_value("SNOWFLAKE_PASSWORD", yaml_path="data_warehouse.snowflake.password")
        self.warehouse = get_config_value("SNOWFLAKE_WAREHOUSE", yaml_path="data_warehouse.snowflake.warehouse")
        self.database = get_config_value("SNOWFLAKE_DATABASE", yaml_path="data_warehouse.snowflake.database")
        self.schema = get_config_value("SNOWFLAKE_SCHEMA", yaml_path="data_warehouse.snowflake.schema")
        self.role = get_config_value("SNOWFLAKE_ROLE", yaml_path="data_warehouse.snowflake.role") # Optional

        self.conn = None
        self._initialized = False

        if not all([self.account, self.user, self.password]):
            logger.error("Snowflake account, user, or password not configured. Connector will not initialize.")
            return

        self._connect()

    def _connect(self):
        try:
            self.conn = snowflake.connector.connect(
                user=self.user,
                password=self.password,
                account=self.account,
                warehouse=self.warehouse,
                database=self.database,
                schema=self.schema,
                role=self.role # Pass role if configured, otherwise it's None and ignored
                # session_parameters={ # Example session parameters
                #     'QUERY_TAG': 'PythonSnowflakeConnector'
                # }
            )
            logger.info(f"Successfully connected to Snowflake account: {self.account}, user: {self.user}")
            self._initialized = True
        except ProgrammingError as db_err: # Errors like invalid credentials, object not found
            logger.error(f"Snowflake ProgrammingError during connection: {db_err}")
            self.conn = None
            self._initialized = False
        except OperationalError as op_err: # Errors like network issues, service unavailable
            logger.error(f"Snowflake OperationalError during connection: {op_err}")
            self.conn = None
            self._initialized = False
        except Exception as e:
            logger.error(f"An unexpected error occurred connecting to Snowflake: {e}", exc_info=True)
            self.conn = None
            self._initialized = False

    def get_connection(self):
        if not self._initialized or not self.conn:
            logger.warning("Snowflake connection not initialized or lost. Attempting to reconnect.")
            self._connect()

        if not self.conn:
            logger.error("Failed to establish Snowflake connection.")
        return self.conn

    def close_connection(self):
        if self.conn:
            try:
                self.conn.close()
                logger.info("Snowflake connection closed.")
            except Exception as e:
                logger.error(f"Error closing Snowflake connection: {e}", exc_info=True)
            finally:
                self.conn = None
                self._initialized = False # Mark as not initialized after closing

    def execute_query(self, query: str, params: Optional[Union[Dict, List]] = None, fetch_one: bool = False, fetch_all: bool = False) -> Optional[Any]:
        """
        Executes a SQL query on Snowflake.
        Args:
            query (str): The SQL query to execute.
            params (Optional[Union[Dict, List]]): Parameters for the query (for binding).
                                                 Use list for %s style, dict for %(name)s style.
            fetch_one (bool): If True, fetches one row.
            fetch_all (bool): If True, fetches all rows.
        Returns:
            Query result (single row, list of rows, or row count) or None on error.
        """
        conn = self.get_connection()
        if not conn:
            return None

        cursor = None
        try:
            cursor = conn.cursor()
            # Snowflake connector typically uses %s for qmark style or %(name)s for pyformat style automatically.
            # For qmark style, params should be a sequence. For pyformat, a dict.
            cursor.execute(query, params)

            if fetch_one:
                return cursor.fetchone()
            if fetch_all:
                return cursor.fetchall()

            # For DML/DDL statements, rowcount might be relevant
            return cursor.rowcount

        except ProgrammingError as db_err:
            logger.error(f"Snowflake ProgrammingError executing query '{query[:100]}...': {db_err}")
            return None
        except OperationalError as op_err:
            logger.error(f"Snowflake OperationalError executing query '{query[:100]}...': {op_err}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error executing query '{query[:100]}...': {e}", exc_info=True)
            return None
        finally:
            if cursor:
                cursor.close()
            # Connection is managed by the class instance, not closed after each query.

# Example Usage (for testing purposes - requires Snowflake account and credentials)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # --- IMPORTANT ---
    # For this example to run, you MUST set the following environment variables:
    # SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD
    # Optionally: SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA, SNOWFLAKE_ROLE
    # e.g., export SNOWFLAKE_USER="your_user" (or set them in config.yaml and ensure it's read)

    logger.info("Attempting to connect to Snowflake (ensure environment variables are set)...")

    # Example: Create a dummy config.yaml for this test if not using env vars
    # Ensure get_config_value can find it, or pass path to it.
    # with open("config.yaml", "w") as f:
    #     f.write("""
    # data_warehouse:
    #   snowflake:
    #     account: "your_account_identifier.region.cloud_provider" # e.g., xy12345.us-east-1
    #     user: "YOUR_SF_USER"
    #     password: "YOUR_SF_PASSWORD"
    #     warehouse: "YOUR_SF_WAREHOUSE" # e.g., COMPUTE_WH
    #     database: "YOUR_SF_DATABASE"   # e.g., MY_DB
    #     schema: "PUBLIC"              # e.g., PUBLIC
    #     # role: "YOUR_SF_ROLE"        # Optional
    # """)

    snowflake_connector = SnowflakeConnector()

    if snowflake_connector._initialized:
        logger.info("SnowflakeConnector initialized successfully.")

        # Test 1: Simple query
        query1 = "SELECT CURRENT_VERSION();"
        version = snowflake_connector.execute_query(query1, fetch_one=True)
        if version:
            logger.info(f"Snowflake version: {version[0]}")

        # Test 2: Query with parameters (example, adjust table/data as per your SF account)
        # query2 = "SELECT * FROM MY_TABLE WHERE MY_COLUMN = %s LIMIT %s;"
        # params2 = ('some_value', 1)
        # results2 = snowflake_connector.execute_query(query2, params2, fetch_all=True)
        # if results2 is not None:
        #     logger.info(f"Query 2 results ({len(results2)} rows):")
        #     for row in results2:
        #         logger.info(row)
        # else:
        #     logger.info("Query 2 returned no results or failed.")

        # Test 3: DML (Example - be careful with DML on actual data)
        # try:
        #     logger.info("Attempting DDL/DML (CREATE TABLE, INSERT, DELETE)...")
        #     snowflake_connector.execute_query("CREATE OR REPLACE TABLE temp_test_table (id INT, name VARCHAR);")
        #     logger.info("temp_test_table created.")
        #     inserted_count = snowflake_connector.execute_query("INSERT INTO temp_test_table (id, name) VALUES (%s, %s), (%s, %s);", (1, 'Test A', 2, 'Test B'))
        #     logger.info(f"Inserted {inserted_count} rows.")

        #     all_temp_data = snowflake_connector.execute_query("SELECT * FROM temp_test_table;", fetch_all=True)
        #     logger.info(f"Data in temp_test_table: {all_temp_data}")

        #     deleted_count = snowflake_connector.execute_query("DELETE FROM temp_test_table WHERE id = %s;", (1,))
        #     logger.info(f"Deleted {deleted_count} row(s).")

        #     snowflake_connector.execute_query("DROP TABLE temp_test_table;")
        #     logger.info("temp_test_table dropped.")
        # except Exception as e:
        #     logger.error(f"Error during DML/DDL test: {e}")


        snowflake_connector.close_connection()
    else:
        logger.error("SnowflakeConnector failed to initialize. Check configurations and credentials.")
        logger.error("Ensure SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD are set as env vars or in a config.yaml accessible by get_config_value.")

    # Clean up dummy config if created
    # if os.path.exists("config.yaml"):
    #     os.remove("config.yaml")
