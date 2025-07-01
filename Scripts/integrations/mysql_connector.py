import pymysql
import logging
from typing import Optional, List, Tuple, Dict, Any, Union

# Assuming MySQLConfig is defined in config.manager:
# from ..config.manager import MySQLConfig

logger = logging.getLogger(__name__)

class MySQLConnector:
    def __init__(self, config: Any): # config should be compatible with MySQLConfig
        self.db_config = {
            "host": getattr(config, 'host', 'localhost'),
            "port": getattr(config, 'port', 3306),
            "user": getattr(config, 'username', 'root'), # Should come from secrets
            "password": getattr(config, 'password', ''),   # Should come from secrets
            "database": getattr(config, 'database', 'rag_db_mysql'),
            "connect_timeout": getattr(config, 'connection_timeout', 10),
            "cursorclass": pymysql.cursors.DictCursor # Return results as dictionaries
        }
        self.connection = None
        # PyMySQL doesn't have a built-in pool like psycopg2's SimpleConnectionPool.
        # For production, a proper pooling library like DBUtils, SQLAlchemy's pool, or a custom one would be needed.
        # For this basic connector, we'll manage a single connection or reconnect.
        try:
            self._connect()
            logger.info(f"MySQL connector initialized for {self.db_config['database']} on {self.db_config['host']}:{self.db_config['port']}")
            self.ping()
        except Exception as e:
            logger.error(f"Failed to initialize MySQL connector: {e}", exc_info=True)
            # self.connection will be None, subsequent calls will try to reconnect or fail.

    def _connect(self):
        """Establishes a connection to MySQL."""
        if self.connection and self.connection.open:
            try:
                self.connection.ping(reconnect=True) # Check if connection is alive, try to reconnect if not
                logger.debug("MySQL connection is active.")
                return
            except pymysql.Error:
                logger.info("MySQL connection lost, attempting to reconnect.")

        try:
            self.connection = pymysql.connect(**self.db_config)
            logger.info(f"Successfully connected to MySQL database '{self.db_config['database']}'.")
        except pymysql.Error as e:
            logger.error(f"Failed to connect to MySQL: {e}", exc_info=True)
            self.connection = None
            raise ConnectionError(f"MySQL connection failed: {e}")


    def close_connection(self):
        """Closes the MySQL connection."""
        if self.connection and self.connection.open:
            try:
                self.connection.close()
                logger.info("MySQL connection closed.")
            except pymysql.Error as e:
                logger.error(f"Error closing MySQL connection: {e}", exc_info=True)
            finally:
                self.connection = None

    def __enter__(self):
        self._connect() # Ensure connection is active
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_connection()

    def ping(self) -> bool:
        """Checks if the database connection is alive."""
        try:
            self._connect() # Ensures connection is attempted/re-established
            if self.connection:
                self.connection.ping()
                logger.info("MySQL connection ping successful.")
                return True
            return False
        except pymysql.Error as e:
            logger.error(f"MySQL connection ping failed: {e}")
            return False

    def execute_query(self, query: str, params: Optional[Union[List, Tuple, Dict]] = None, fetch_one: bool = False, fetch_all: bool = False, commit: bool = False) -> Optional[Any]:
        """
        Executes a SQL query.

        Args:
            query (str): The SQL query to execute.
            params (Optional[Union[List, Tuple, Dict]]): Parameters for the query.
            fetch_one (bool): If True, fetches one row (as dict).
            fetch_all (bool): If True, fetches all rows (as list of dicts).
            commit (bool): If True, commits the transaction (for INSERT, UPDATE, DELETE).

        Returns:
            Optional[Any]: Query result or rowcount.
        """
        result = None
        try:
            self._connect() # Ensure connection
            if not self.connection:
                raise ConnectionError("MySQL not connected.")

            with self.connection.cursor() as cursor:
                cursor.execute(query, params)

                if fetch_one:
                    result = cursor.fetchone()
                elif fetch_all:
                    result = cursor.fetchall()
                else:
                    result = cursor.rowcount

                if commit:
                    self.connection.commit()
            return result
        except pymysql.Error as e:
            if self.connection and not commit:
                try:
                    self.connection.rollback()
                except Exception as rb_err:
                    logger.error(f"Error during MySQL rollback: {rb_err}")
            logger.error(f"Error executing MySQL query '{query[:100]}...': {e}", exc_info=True)
            raise
        # No finally block to close connection here, as it's managed per instance or via context manager.
        # For a pooled setup, connections would be released here.

    # --- Placeholder methods for common operations ---
    def insert_data(self, table_name: str, data: Dict[str, Any]) -> Optional[int]:
        """Placeholder for inserting data. Returns lastrowid if possible."""
        # Example:
        # columns = ", ".join(data.keys())
        # placeholders = ", ".join(["%s"] * len(data))
        # query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        # self.execute_query(query, list(data.values()), commit=True)
        # return self.connection.insert_id() if self.connection else None
        logger.info(f"Placeholder: insert_data into {table_name} with {data}")
        return 1 # Simulate an ID or row count

    def get_data_by_id(self, table_name: str, item_id: Any, id_column: str = "id") -> Optional[Dict[str, Any]]:
        """Placeholder for fetching data by ID."""
        # Example:
        # query = f"SELECT * FROM {table_name} WHERE {id_column} = %s"
        # return self.execute_query(query, (item_id,), fetch_one=True)
        logger.info(f"Placeholder: get_data_by_id from {table_name} for ID {item_id}")
        return {"id": item_id, "data": "sample mysql data"}


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # This example requires a running MySQL instance and proper credentials.
    class MockMySQLConfig: # Replace with actual config loading in app
        host = "localhost"
        port = 3306
        username = "your_mysql_user"  # Replace
        password = "your_mysql_password" # Replace
        database = "your_mysql_db"    # Replace
        connection_timeout = 10

    test_mysql_config = MockMySQLConfig()

    # Check if placeholder credentials are used
    if test_mysql_config.username == "your_mysql_user":
        print("MySQL example skipped: Please update mock credentials in the script for testing.")
    else:
        try:
            with MySQLConnector(config=test_mysql_config) as mysql_conn:
                if mysql_conn.ping():
                    print("Successfully connected to MySQL.")

                    # Example: Create a test table (idempotent)
                    create_table_query = """
                    CREATE TABLE IF NOT EXISTS test_mysql_items (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    ) ENGINE=InnoDB;
                    """
                    mysql_conn.execute_query(create_table_query, commit=True) # commit for DDL
                    print("Test table 'test_mysql_items' ensured.")

                    # Example: Insert data
                    item_name = f"MySQLTestItem_{int(time.time())}"
                    # PyMySQL uses %s for placeholders
                    insert_query = "INSERT INTO test_mysql_items (name) VALUES (%s)"
                    mysql_conn.execute_query(insert_query, (item_name,), commit=True)
                    # Get last inserted ID (specific to PyMySQL and connection state)
                    last_id_query = "SELECT LAST_INSERT_ID()"
                    last_id_result = mysql_conn.execute_query(last_id_query, fetch_one=True)
                    inserted_id = last_id_result['LAST_INSERT_ID()'] if last_id_result else None

                    if inserted_id:
                        print(f"Inserted item with ID: {inserted_id} and name: {item_name}")

                        # Example: Fetch data
                        select_query = "SELECT id, name, created_at FROM test_mysql_items WHERE id = %s;"
                        row = mysql_conn.execute_query(select_query, (inserted_id,), fetch_one=True)
                        if row:
                            print(f"Fetched item: {row}")

                    # Example: Fetch all
                    all_items = mysql_conn.execute_query("SELECT name FROM test_mysql_items LIMIT 5;", fetch_all=True)
                    print(f"Fetched up to 5 items: {all_items}")

        except ConnectionError as ce:
            print(f"MySQL Connection error: {ce}")
        except Exception as e:
            print(f"An error occurred with MySQL: {e}", exc_info=True)
