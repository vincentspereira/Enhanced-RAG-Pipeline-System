import sqlite3
import logging
from typing import Optional, List, Tuple, Dict, Any, Union
from pathlib import Path

# Assuming SQLiteConfig is defined in config.manager:
# from ..config.manager import SQLiteConfig

logger = logging.getLogger(__name__)

class SQLiteConnector:
    def __init__(self, config: Any): # config should be compatible with SQLiteConfig
        self.db_path = Path(getattr(config, 'db_path', 'rag_sqlite.db'))
        self.timeout = getattr(config, 'timeout', 5.0) # Default timeout in seconds

        # Ensure parent directory for db_path exists if it's not in-memory
        if str(self.db_path) != ":memory:":
            try:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not create parent directory for SQLite DB at {self.db_path.parent}: {e}. This might fail if path is complex and non-existent.")

        self.connection: Optional[sqlite3.Connection] = None
        # sqlite3 connections are typically lightweight. Pooling is less common for SQLite
        # unless dealing with very high concurrency in a threaded environment, which needs careful setup.
        # For many use cases, opening/closing connections or keeping one open is fine.
        try:
            self._connect()
            logger.info(f"SQLite connector initialized for database at '{self.db_path}'.")
            if self.ping(): # Initial ping
                 logger.info(f"Successfully connected to SQLite database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite connector for {self.db_path}: {e}", exc_info=True)
            # self.connection will be None

    def _connect(self):
        """Establishes a connection to SQLite. Creates the DB file if it doesn't exist (unless :memory:)."""
        if self.connection: # Basic check if a connection object exists
            try:
                # Check if connection is still usable by executing a simple pragma
                self.connection.execute("PRAGMA user_version").fetchone()
                logger.debug("SQLite connection is active.")
                return
            except sqlite3.Error: # Catches errors like ProgrammingError if connection is closed
                logger.info("SQLite connection lost or closed, attempting to reconnect.")
                self.connection = None # Force re-creation

        try:
            self.connection = sqlite3.connect(str(self.db_path), timeout=self.timeout)
            self.connection.row_factory = sqlite3.Row # Access columns by name
            logger.debug(f"Successfully connected to SQLite database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to SQLite at '{self.db_path}': {e}", exc_info=True)
            self.connection = None
            raise ConnectionError(f"SQLite connection failed for '{self.db_path}': {e}")

    def close_connection(self):
        """Closes the SQLite connection."""
        if self.connection:
            try:
                self.connection.close()
                logger.info(f"SQLite connection to '{self.db_path}' closed.")
            except sqlite3.Error as e:
                logger.error(f"Error closing SQLite connection: {e}", exc_info=True)
            finally:
                self.connection = None

    def __enter__(self):
        self._connect() # Ensure connection is active
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_connection()

    def ping(self) -> bool:
        """Checks if the database connection is alive by executing a simple pragma."""
        try:
            self._connect() # Ensure connection
            if self.connection:
                self.connection.execute("PRAGMA quick_check").fetchone() # quick_check is a bit more thorough than just "SELECT 1"
                logger.info(f"SQLite connection ping successful for '{self.db_path}'.")
                return True
            return False
        except sqlite3.Error as e:
            logger.error(f"SQLite connection ping failed for '{self.db_path}': {e}")
            return False

    def execute_query(self, query: str, params: Optional[Union[List, Tuple]] = None, fetch_one: bool = False, fetch_all: bool = False, commit: bool = False) -> Optional[Any]:
        """
        Executes a SQL query.

        Args:
            query (str): The SQL query to execute.
            params (Optional[Union[List, Tuple]]): Parameters for the query.
            fetch_one (bool): If True, fetches one row (as sqlite3.Row, acts like a dict).
            fetch_all (bool): If True, fetches all rows (as list of sqlite3.Row).
            commit (bool): If True, commits the transaction (for INSERT, UPDATE, DELETE).

        Returns:
            Optional[Any]: Query result (single row, all rows, or lastrowid/rowcount).
        """
        result = None
        params = params or []
        try:
            self._connect()
            if not self.connection:
                raise ConnectionError("SQLite not connected.")

            cursor = self.connection.cursor()
            cursor.execute(query, params)

            if fetch_one:
                row = cursor.fetchone()
                result = dict(row) if row else None # Convert sqlite3.Row to dict
            elif fetch_all:
                rows = cursor.fetchall()
                result = [dict(row) for row in rows] # Convert list of sqlite3.Row to list of dicts
            else:
                # For DML, lastrowid is useful for inserts, rowcount for updates/deletes
                result = cursor.lastrowid if cursor.lastrowid is not None and cursor.lastrowid > 0 else cursor.rowcount

            if commit:
                self.connection.commit()

            return result
        except sqlite3.Error as e:
            if self.connection and not commit: # SQLite auto-rolls back on error if commit isn't called for the transaction
                pass # No explicit rollback needed typically unless managing transactions manually
            logger.error(f"Error executing SQLite query '{query[:100]}...': {e}", exc_info=True)
            raise
        # Connection is kept open by default for the instance lifecycle or managed by context manager.
        # cursor.close() is implicitly called when 'with self.connection.cursor() as cursor:' exits, if that pattern was used.
        # Here, cursor is method-scoped, so it's fine.

    # --- Placeholder methods for common operations ---
    def insert_data(self, table_name: str, data: Dict[str, Any]) -> Optional[int]:
        """Inserts data and returns the last inserted row ID."""
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        logger.info(f"Inserting data into SQLite table {table_name}: {data}")
        return self.execute_query(query, list(data.values()), commit=True) # Returns lastrowid

    def get_data_by_id(self, table_name: str, item_id: Any, id_column: str = "id") -> Optional[Dict[str, Any]]:
        """Fetches data by ID."""
        query = f"SELECT * FROM {table_name} WHERE {id_column} = ?"
        logger.info(f"Fetching data by ID from SQLite table {table_name} for ID {item_id}")
        return self.execute_query(query, (item_id,), fetch_one=True)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Example usage with an in-memory SQLite database
    class MockSQLiteConfig:
        db_path: str = ":memory:" # In-memory database for testing
        # db_path: str = "test_sqlite_connector.db" # File-based database for testing
        timeout: int = 5

    test_sqlite_config = MockSQLiteConfig()

    # Ensure the test DB file is clean if file-based
    # if str(test_sqlite_config.db_path) != ":memory:" and Path(test_sqlite_config.db_path).exists():
    #     Path(test_sqlite_config.db_path).unlink()

    try:
        with SQLiteConnector(config=test_sqlite_config) as sqlite_conn:
            if sqlite_conn.ping():
                print(f"Successfully connected to SQLite DB at '{sqlite_conn.db_path}'.")

                create_table_query = """
                CREATE TABLE IF NOT EXISTS test_sqlite_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
                sqlite_conn.execute_query(create_table_query, commit=True)
                print("Test table 'test_sqlite_items' ensured.")

                # Insert data using specific method
                item_name = f"SQLiteItem_{int(time.time())}"
                inserted_id = sqlite_conn.insert_data("test_sqlite_items", {"name": item_name, "value": 123.45})
                print(f"Inserted item with ID: {inserted_id} and name: {item_name}")

                if inserted_id:
                    # Fetch data using specific method
                    fetched_item = sqlite_conn.get_data_by_id("test_sqlite_items", inserted_id)
                    print(f"Fetched item by ID {inserted_id}: {fetched_item}")

                # Insert multiple
                sqlite_conn.execute_query("INSERT INTO test_sqlite_items (name, value) VALUES (?, ?)", ("Another Item", 54.321), commit=True)
                sqlite_conn.execute_query("INSERT INTO test_sqlite_items (name, value) VALUES (?, ?)", ("Third Item", 99.0), commit=True)


                # Fetch all
                all_items = sqlite_conn.execute_query("SELECT id, name, value FROM test_sqlite_items;", fetch_all=True)
                print(f"Fetched all items ({len(all_items)}):")
                for item in all_items:
                    print(item)
            else:
                print(f"Failed to connect to SQLite DB at {test_sqlite_config.db_path}")

    except ConnectionError as ce:
        print(f"SQLite Connection error: {ce}")
    except Exception as e:
        import time # Make sure time is imported for the example name
        print(f"An error occurred with SQLite: {e}", exc_info=True)
    finally:
        # Cleanup test DB file if created
        # if str(test_sqlite_config.db_path) != ":memory:" and Path(test_sqlite_config.db_path).exists():
        #     Path(test_sqlite_config.db_path).unlink()
        #     print(f"Cleaned up test SQLite DB file: {test_sqlite_config.db_path}")
        pass
