import os
import sqlite3
import logging

# Configure logger
logger = logging.getLogger(__name__)

# Import config loader
try:
    from Scripts.utils.config_loader import get_config_value
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        from utils.config_loader import get_config_value
    except ImportError as e:
        logger.error(f"Critical: Failed to import get_config_value for SQLiteConnector. Error: {e}")
        def get_config_value(env_var_name, yaml_path=None, default=None): # Basic fallback
            return os.getenv(env_var_name, default)

class SQLiteConnector:
    _connections = {} # Class-level dictionary to store connections by db_path

    def __init__(self, db_path: Optional[str] = None):
        # If db_path is None, it will try to load from config or use default.
        # This allows creating an instance without immediately knowing the path,
        # or to manage multiple SQLite DBs by creating separate instances with specific paths.

        self.db_path_config_key = "SQLITE_DB_PATH" # Env var name
        self.db_path_yaml_key = "database.sqlite.path" # Path in config.yaml
        self.default_db_path = "data/sqlite.db" # Default path if nothing configured

        if db_path:
            self.db_path = db_path
        else:
            self.db_path = get_config_value(
                self.db_path_config_key,
                yaml_path=self.db_path_yaml_key,
                default=self.default_db_path
            )

        # Ensure the directory for the SQLite file exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            try:
                os.makedirs(db_dir, exist_ok=True)
                logger.info(f"Created directory for SQLite database: {db_dir}")
            except OSError as e:
                logger.error(f"Failed to create directory {db_dir} for SQLite database: {e}")
                # Depending on strictness, might raise an error here or allow connection attempt to fail later

        self.conn = None
        self._initialized = False
        self._connect() # Attempt to connect on init

    def _connect(self):
        # SQLite connections are typically lightweight.
        # This connector will maintain one connection per instance (and thus per db_path).
        # For very high concurrency, advanced pooling might be needed, but sqlite3 itself
        # handles a good degree of concurrency with WAL mode, etc.
        if self.db_path in SQLiteConnector._connections and SQLiteConnector._connections[self.db_path]:
            self.conn = SQLiteConnector._connections[self.db_path]
            logger.info(f"Reusing existing SQLite connection for {self.db_path}")
            self._initialized = True
            return

        try:
            # `check_same_thread=False` is often needed for web apps where different threads might use the connection.
            # However, it requires careful handling of shared resources if any.
            # For a simple connector, it's a common setting.
            # Consider `timeout` for busy databases.
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10) # 10 second timeout
            self.conn.row_factory = sqlite3.Row # Access columns by name
            SQLiteConnector._connections[self.db_path] = self.conn
            logger.info(f"SQLite connection established to {self.db_path}")
            self._initialized = True
        except sqlite3.Error as e:
            logger.error(f"Error connecting to SQLite database at {self.db_path}: {e}", exc_info=True)
            self.conn = None
            self._initialized = False

    def get_connection(self):
        if not self.conn or not self._initialized:
            logger.warning(f"SQLite connection to {self.db_path} not initialized or lost. Attempting to reconnect.")
            self._connect() # Try to reconnect

        if not self.conn: # If reconnect failed
             logger.error(f"Failed to establish SQLite connection to {self.db_path}.")
        return self.conn

    def close_connection(self):
        # This method will close the connection associated with this specific instance's db_path.
        if self.db_path in SQLiteConnector._connections:
            conn_to_close = SQLiteConnector._connections.pop(self.db_path) # Remove from shared pool
            if conn_to_close:
                try:
                    conn_to_close.close()
                    logger.info(f"SQLite connection to {self.db_path} closed.")
                except sqlite3.Error as e:
                    logger.error(f"Error closing SQLite connection to {self.db_path}: {e}", exc_info=True)
        if self.conn: # Also clear instance variable
            self.conn = None
        self._initialized = False

    @classmethod
    def close_all_connections(cls):
        # Class method to close all managed SQLite connections
        for path, conn in list(cls._connections.items()): # Iterate over a copy for safe removal
            try:
                if conn:
                    conn.close()
                logger.info(f"Closed SQLite connection to {path} (via close_all).")
            except sqlite3.Error as e:
                logger.error(f"Error closing SQLite connection to {path} during close_all: {e}", exc_info=True)
            del cls._connections[path]


    def execute_query(self, query, params=None, fetch_one=False, fetch_all=False, commit=False):
        conn = self.get_connection()
        if not conn:
            return None

        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute(query, params or ()) # params must be a tuple or sequence

            if commit:
                conn.commit()
                logger.info("Query executed and committed successfully.")
                return cursor.lastrowid if cursor.lastrowid is not None else cursor.rowcount

            if fetch_one:
                return cursor.fetchone() # Returns a Row object or None
            if fetch_all:
                return cursor.fetchall() # Returns a list of Row objects

            return cursor.rowcount # For SELECTs not fetched, or other non-committing DML

        except sqlite3.Error as e:
            logger.error(f"Error executing SQLite query: {query} with params: {params}. Error: {e}", exc_info=True)
            # SQLite automatically rolls back on error if a transaction was started implicitly or explicitly
            # and not committed. No explicit rollback needed here unless managing transactions manually.
            return None
        finally:
            if cursor:
                cursor.close()
            # Connection is kept open for the lifetime of the SQLiteConnector instance or until close_connection / close_all_connections.

# Example Usage (for testing purposes)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Test with default path
    logger.info("--- Testing with default DB path ---")
    connector1 = SQLiteConnector() # Uses default path from config or 'data/sqlite.db'

    if not connector1._initialized:
        logger.error("SQLiteConnector (connector1) not initialized. Exiting test.")
        exit()

    create_table_q = """
    CREATE TABLE IF NOT EXISTS sqlite_test_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        quantity INTEGER
    );
    """
    connector1.execute_query(create_table_q, commit=True)
    logger.info("Table 'sqlite_test_items' created or already exists in default DB.")

    insert_q = "INSERT INTO sqlite_test_items (name, quantity) VALUES (?, ?)"
    last_id = connector1.execute_query(insert_q, ("SQLite Item A", 20), commit=True)
    logger.info(f"Inserted item, lastrowid: {last_id}")

    select_one_q = "SELECT * FROM sqlite_test_items WHERE name = ?"
    item = connector1.execute_query(select_one_q, ("SQLite Item A",), fetch_one=True)
    if item:
        logger.info(f"Selected item: ID={item['id']}, Name={item['name']}, Quantity={item['quantity']}")

    select_all_q = "SELECT * FROM sqlite_test_items"
    all_items = connector1.execute_query(select_all_q, fetch_all=True)
    if all_items:
        logger.info(f"All items in default DB: [{', '.join([i['name'] for i in all_items])}]")

    # Test with a specific, different path
    logger.info("\n--- Testing with specific DB path 'data/another_sqlite.db' ---")
    custom_db_path = "data/another_sqlite.db"
    if os.path.exists(custom_db_path): # Clean up from previous test if any
        os.remove(custom_db_path)

    connector2 = SQLiteConnector(db_path=custom_db_path)
    if not connector2._initialized:
        logger.error("SQLiteConnector (connector2) not initialized. Exiting test.")
        exit()

    connector2.execute_query(create_table_q, commit=True) # Create table in the new DB
    logger.info("Table 'sqlite_test_items' created or already exists in another_sqlite.db.")

    last_id_2 = connector2.execute_query(insert_q, ("SQLite Item B in other DB", 30), commit=True)
    logger.info(f"Inserted item in another_sqlite.db, lastrowid: {last_id_2}")
    item2 = connector2.execute_query(select_one_q, ("SQLite Item B in other DB",), fetch_one=True)
    if item2:
        logger.info(f"Selected item from another_sqlite.db: {item2['name']}")

    # Verify that connector1 still points to its own DB and doesn't see Item B
    item_a_still_there = connector1.execute_query(select_one_q, ("SQLite Item A",), fetch_one=True)
    item_b_in_conn1 = connector1.execute_query(select_one_q, ("SQLite Item B in other DB",), fetch_one=True)
    logger.info(f"Connector 1 still sees Item A: {bool(item_a_still_there)}")
    logger.info(f"Connector 1 does NOT see Item B: {not bool(item_b_in_conn1)}")


    # Clean up: Close all connections
    # connector1.close_connection() # Closes connection to default_db_path
    # connector2.close_connection() # Closes connection to custom_db_path
    SQLiteConnector.close_all_connections()
    logger.info("All SQLite connections closed.")

    # Optional: remove test databases
    # if os.path.exists(connector1.db_path): os.remove(connector1.db_path)
    # if os.path.exists(custom_db_path): os.remove(custom_db_path)
