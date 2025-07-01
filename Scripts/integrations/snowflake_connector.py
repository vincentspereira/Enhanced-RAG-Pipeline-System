import snowflake.connector
from snowflake.connector.errors import ProgrammingError, DatabaseError
import logging
from typing import Optional, List, Tuple, Dict, Any, Union

# Assuming SnowflakeConfig is defined in config.manager:
# from ..config.manager import SnowflakeConfig

logger = logging.getLogger(__name__)

class SnowflakeConnector:
    def __init__(self, config: Any): # config should be compatible with SnowflakeConfig
        self.account = getattr(config, 'account', None)
        self.username = getattr(config, 'username', None) # Should come from secrets
        self.password = getattr(config, 'password', None) # Should come from secrets
        self.warehouse = getattr(config, 'warehouse', None)
        self.database = getattr(config, 'database', None)
        self.schema_name = getattr(config, 'schema_name', 'PUBLIC') # schema is often a keyword
        self.role = getattr(config, 'role', None)
        # self.authenticator = getattr(config, 'authenticator', None) # For other auth methods

        if not self.account or not self.username: # Password might be handled by other auth methods
            logger.error("Snowflake account and username are required for connection.")
            raise ValueError("Snowflake account and username must be provided in the configuration.")

        self.connection: Optional[snowflake.connector.SnowflakeConnection] = None
        try:
            self._connect()
            logger.info(f"Snowflake connector initialized for account '{self.account}', user '{self.username}'.")
            if self.ping():
                logger.info("Successfully connected to Snowflake.")
        except Exception as e:
            logger.error(f"Failed to initialize Snowflake connector: {e}", exc_info=True)
            # self.connection will be None

    def _connect(self):
        """Establishes or verifies a connection to Snowflake."""
        if self.connection and not self.connection.is_closed():
            logger.debug("Snowflake connection is active.")
            return

        try:
            conn_params = {
                "user": self.username,
                "account": self.account,
                "warehouse": self.warehouse,
                "database": self.database,
                "schema": self.schema_name, # 'schema' is the kwarg for snowflake connector
                "role": self.role
            }
            # Only add password if it's provided (to support authenticator methods like key-pair)
            if self.password:
                conn_params["password"] = self.password
            # if self.authenticator:
            #     conn_params["authenticator"] = self.authenticator

            # Remove None values from conn_params
            conn_params = {k: v for k, v in conn_params.items() if v is not None}

            self.connection = snowflake.connector.connect(**conn_params)
            logger.info(f"Successfully connected to Snowflake account '{self.account}'.")
        except DatabaseError as e:
            logger.error(f"Failed to connect to Snowflake: {e}", exc_info=True)
            self.connection = None
            raise ConnectionError(f"Snowflake connection failed: {e}")
        except Exception as e: # Catch other potential errors
            logger.error(f"An unexpected error occurred while connecting to Snowflake: {e}", exc_info=True)
            self.connection = None
            raise ConnectionError(f"Snowflake connection failed unexpectedly: {e}")


    def close_connection(self):
        """Closes the Snowflake connection."""
        if self.connection and not self.connection.is_closed():
            try:
                self.connection.close()
                logger.info("Snowflake connection closed.")
            except DatabaseError as e:
                logger.error(f"Error closing Snowflake connection: {e}", exc_info=True)
            finally:
                self.connection = None

    def __enter__(self):
        self._connect() # Ensure connection is active
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_connection()

    def ping(self) -> bool:
        """Checks if the database connection is alive by executing a simple query."""
        try:
            self._connect() # Ensure connection
            if self.connection:
                with self.connection.cursor() as cursor:
                    cursor.execute("SELECT 1;")
                logger.info("Snowflake connection ping successful.")
                return True
            return False
        except DatabaseError as e:
            logger.error(f"Snowflake connection ping failed: {e}")
            return False
        except Exception as e: # Catch if _connect itself fails
            logger.error(f"Snowflake ping failed due to connection issue: {e}")
            return False


    def execute_query(self, query: str, params: Optional[Union[List, Tuple]] = None, fetch_one: bool = False, fetch_all: bool = False, commit_needed: bool = False) -> Optional[Any]:
        """
        Executes a SQL query. Snowflake auto-commits DML by default unless in an explicit transaction.
        Use `commit_needed=True` if you've started a transaction explicitly and want to commit.

        Args:
            query (str): The SQL query to execute.
            params (Optional[Union[List, Tuple]]): Parameters for the query (qmark or numeric style).
            fetch_one (bool): If True, fetches one row (as dict).
            fetch_all (bool): If True, fetches all rows (as list of dicts).
            commit_needed (bool): If True, explicitly commits the transaction.

        Returns:
            Optional[Any]: Query result or rowcount.
        """
        result = None
        try:
            self._connect()
            if not self.connection:
                raise ConnectionError("Snowflake not connected.")

            # Snowflake connector's DictCursor returns results as dictionaries
            with self.connection.cursor(snowflake.connector.DictCursor) as cursor:
                cursor.execute(query, params)

                if fetch_one:
                    result = cursor.fetchone()
                elif fetch_all:
                    result = cursor.fetchall()
                else: # For DML (INSERT, UPDATE, DELETE), get rowcount
                    result = cursor.rowcount

                if commit_needed: # Only commit if explicitly told to
                    self.connection.commit()
            return result
        except ProgrammingError as e: # Specific error type from snowflake connector
            if self.connection and commit_needed == False: # Check if rollback is meaningful
                 try: self.connection.rollback()
                 except Exception as rb_err: logger.error(f"Error during Snowflake rollback: {rb_err}")
            logger.error(f"Error executing Snowflake query '{query[:100]}...': {e}", exc_info=True)
            raise
        except DatabaseError as e:
            logger.error(f"DatabaseError executing Snowflake query '{query[:100]}...': {e}", exc_info=True)
            raise
        # Connection is kept open by instance or context manager

    # --- Placeholder methods for common operations ---
    def load_data(self, table_name: str, file_path: str, file_format: str = "CSV") -> None:
        """Placeholder for loading data from a file into Snowflake (e.g., using PUT and COPY INTO)."""
        logger.info(f"Placeholder: load_data from {file_path} (format: {file_format}) into Snowflake table {table_name}")
        # Example steps:
        # 1. PUT file to Snowflake stage: self.execute_query(f"PUT file://{file_path} @%{table_name}_stage;")
        # 2. COPY INTO table: self.execute_query(f"COPY INTO {table_name} FROM @%{table_name}_stage FILE_FORMAT = (TYPE = '{file_format}' ...);")
        pass

    def get_table_schema(self, table_name: str) -> Optional[List[Dict[str, Any]]]:
        """Placeholder for fetching table schema."""
        logger.info(f"Placeholder: get_table_schema for Snowflake table {table_name}")
        # Example: query = f"DESCRIBE TABLE {table_name};"
        # return self.execute_query(query, fetch_all=True)
        return [{"name": "column_a", "type": "VARCHAR"}]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    import time

    class MockSnowflakeConfig:
        account = os.getenv("SNOWFLAKE_TEST_ACCOUNT")
        username = os.getenv("SNOWFLAKE_TEST_USER")
        password = os.getenv("SNOWFLAKE_TEST_PASSWORD")
        warehouse = os.getenv("SNOWFLAKE_TEST_WAREHOUSE")
        database = os.getenv("SNOWFLAKE_TEST_DATABASE")
        schema_name = os.getenv("SNOWFLAKE_TEST_SCHEMA", "PUBLIC")
        role = os.getenv("SNOWFLAKE_TEST_ROLE")

    test_sf_config = MockSnowflakeConfig()

    if not all([test_sf_config.account, test_sf_config.username, test_sf_config.password, test_sf_config.warehouse, test_sf_config.database]):
        print("Snowflake example skipped: Set SNOWFLAKE_TEST_* environment variables for testing.")
    else:
        sf_connector = None
        try:
            # Using context manager for connect/close
            with SnowflakeConnector(config=test_sf_config) as sf_conn:
                if sf_conn.ping():
                    print("Successfully connected to Snowflake.")

                    # Use current DB and Schema
                    sf_conn.execute_query(f"USE WAREHOUSE {test_sf_config.warehouse};")
                    sf_conn.execute_query(f"USE DATABASE {test_sf_config.database};")
                    sf_conn.execute_query(f"USE SCHEMA {test_sf_config.schema_name};")
                    print(f"Using warehouse: {test_sf_config.warehouse}, database: {test_sf_config.database}, schema: {test_sf_config.schema_name}")

                    table = f"test_sf_items_{int(time.time())}"
                    create_ddl = f"CREATE TABLE IF NOT EXISTS {table} (id INT, name VARCHAR, value FLOAT);"
                    sf_conn.execute_query(create_ddl) # DDLs are auto-committed
                    print(f"Table '{table}' ensured.")

                    insert_dml = f"INSERT INTO {table} (id, name, value) VALUES (%s, %s, %s), (%s, %s, %s);"
                    row_count = sf_conn.execute_query(insert_dml, (1, 'Item A', 10.5, 2, 'Item B', 20.3))
                    print(f"Inserted {row_count} rows.") # Should be 2 for Snowflake INSERT

                    query_result = sf_conn.execute_query(f"SELECT name, value FROM {table} WHERE id = %s;", (1,), fetch_one=True)
                    print(f"Fetched one row (id=1): {query_result}")

                    all_results = sf_conn.execute_query(f"SELECT * FROM {table};", fetch_all=True)
                    print(f"Fetched all rows ({len(all_results)}):")
                    for r in all_results:
                        print(r)

                    sf_conn.execute_query(f"DROP TABLE {table};")
                    print(f"Dropped table '{table}'.")
                else:
                    print("Failed to ping Snowflake.")

        except ConnectionError as ce:
            print(f"Snowflake Connection error: {ce}")
        except DatabaseError as de:
            print(f"Snowflake DatabaseError: {de}")
        except Exception as e:
            print(f"An error occurred with Snowflake: {e}", exc_info=True)
