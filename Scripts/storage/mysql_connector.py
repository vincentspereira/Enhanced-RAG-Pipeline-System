import os
import pymysql
from pymysql.cursors import DictCursor # Example: make results dict-like
from dbutils.pooled_db import PooledDB # For connection pooling
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
        logger.error(f"Critical: Failed to import get_config_value for MySQLConnector. Error: {e}")
        def get_config_value(env_var_name, yaml_path=None, default=None): # Basic fallback
            return os.getenv(env_var_name, default)

class MySQLConnector:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MySQLConnector, cls).__new__(cls)
        return cls._instance

    def __init__(self, min_conn=1, max_conn=5, cursorclass=DictCursor):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.db_host = get_config_value("MYSQL_HOST", yaml_path="database.mysql.host", default="localhost")
        self.db_port = int(get_config_value("MYSQL_PORT", yaml_path="database.mysql.port", default=3306))
        self.db_user = get_config_value("MYSQL_USER", yaml_path="database.mysql.user", default="root")
        self.db_password = get_config_value("MYSQL_PASSWORD", yaml_path="database.mysql.password", default="")
        self.db_name = get_config_value("MYSQL_DB", yaml_path="database.mysql.dbname", default="mydatabase")
        self.db_charset = get_config_value("MYSQL_CHARSET", yaml_path="database.mysql.charset", default="utf8mb4")

        self.min_conn = int(get_config_value("MYSQL_MIN_CONN", default=min_conn))
        self.max_conn = int(get_config_value("MYSQL_MAX_CONN", default=max_conn))
        self.cursorclass = cursorclass # e.g., pymysql.cursors.DictCursor to get results as dicts

        self.pool = None
        self._initialized = False

        try:
            # Using dbutils.pooled_db for connection pooling with PyMySQL
            self.pool = PooledDB(
                creator=pymysql, # Module to use for creating connections
                mincached=self.min_conn,
                maxcached=self.max_conn,
                host=self.db_host,
                port=self.db_port,
                user=self.db_user,
                password=self.db_password,
                database=self.db_name,
                charset=self.db_charset,
                cursorclass=self.cursorclass,
                # autocommit=False # Default is False, explicit commit needed
                # connect_timeout=10 # Example
            )
            if self.pool:
                # Test getting a connection to ensure pool is working
                conn_test = self.get_connection()
                if conn_test:
                    self.release_connection(conn_test)
                    logger.info(f"MySQL connection pool created successfully for {self.db_user}@{self.db_host}:{self.db_port}/{self.db_name}")
                    self._initialized = True
                else:
                    logger.error("Failed to get a test connection from the MySQL pool.")
                    self.pool = None # Mark pool as unusable
            else:
                logger.error("Failed to create MySQL connection pool.")
        except Exception as error:
            logger.error(f"Error while connecting to MySQL or creating pool: {error}", exc_info=True)
            self.pool = None

    def get_connection(self):
        if not self.pool or not self._initialized:
            logger.error("MySQL Connection pool is not initialized.")
            return None
        try:
            return self.pool.connection() # PooledDB method to get a connection
        except Exception as error:
            logger.error(f"Error getting MySQL connection from pool: {error}", exc_info=True)
            return None

    def release_connection(self, conn):
        if not self.pool or not self._initialized:
            logger.warning("MySQL Connection pool is not initialized. Cannot release connection.")
            return
        if conn:
            try:
                conn.close() # For PooledDB, closing the connection returns it to the pool
            except Exception as error:
                logger.error(f"Error releasing MySQL connection to pool: {error}", exc_info=True)

    def execute_query(self, query, params=None, fetch_one=False, fetch_all=False, commit=False):
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            if not conn:
                return None

            cursor = conn.cursor()
            cursor.execute(query, params)

            if commit:
                conn.commit()
                logger.info("Query executed and committed successfully.")
                return cursor.rowcount # For DML statements, rowcount is often useful

            if fetch_one:
                return cursor.fetchone()
            if fetch_all:
                return cursor.fetchall()

            # If not fetching and not committing (e.g. SELECT without fetch, or part of a transaction)
            return cursor.rowcount # Or True to indicate success for non-fetching DQL

        except Exception as error:
            logger.error(f"Error executing MySQL query: {query} with params: {params}. Error: {error}", exc_info=True)
            if conn: # Rollback if an error occurs during a transaction that should be committed
                try:
                    if commit: # Only rollback if commit was intended but failed
                         conn.rollback()
                         logger.info("Transaction rolled back due to error.")
                except Exception as rb_error:
                    logger.error(f"Error during MySQL rollback: {rb_error}")
            return None
        finally:
            # Cursor is closed automatically when connection is closed for PooledDB's shared connections
            # if cursor:
            #     cursor.close()
            if conn:
                self.release_connection(conn)

    def close_pool(self):
        if self.pool and self._initialized:
            try:
                self.pool.close() # Closes all connections in the pool
                logger.info("MySQL connection pool closed.")
                self._initialized = False
            except Exception as error:
                logger.error(f"Error closing MySQL connection pool: {error}", exc_info=True)
        self.pool = None


# Example Usage (for testing purposes)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Set environment variables for testing if not using a config file or actual env vars
    # Ensure your MySQL server is running and accessible with these credentials.
    # os.environ["MYSQL_HOST"] = "localhost"
    # os.environ["MYSQL_USER"] = "youruser"
    # os.environ["MYSQL_PASSWORD"] = "yourpassword"
    # os.environ["MYSQL_DB"] = "yourtestdb"
    # os.environ["MYSQL_PORT"] = "3306"

    mysql_connector = MySQLConnector()

    if not mysql_connector.pool or not mysql_connector._initialized:
        logger.error("MySQLConnector not initialized. Exiting test.")
        exit()

    # Test 1: Create a table
    create_table_query = """
    CREATE TABLE IF NOT EXISTS mysql_test_items (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        quantity INT
    ) ENGINE=InnoDB;
    """
    # DDL statements are often auto-committed or don't need explicit commit depending on server/session settings.
    # Forcing commit=True here for clarity, though it might not always be necessary for DDL.
    if mysql_connector.execute_query(create_table_query, commit=True) is not None: # Check for non-None, as rowcount for DDL can be 0
        logger.info("Table 'mysql_test_items' created or already exists.")

        # Test 2: Insert data
        # Using %s for placeholders is standard for PyMySQL
        insert_query = "INSERT INTO mysql_test_items (name, quantity) VALUES (%s, %s)"
        # PyMySQL's lastrowid is available on the cursor after an INSERT with AUTO_INCREMENT
        # but execute_query as written returns rowcount for committed DML.
        # To get lastrowid, one might need to expose the cursor or modify execute_query.
        # For now, we'll just check rowcount.
        inserted_rows = mysql_connector.execute_query(insert_query, ("MySQL Test Item 1", 15), commit=True)
        if inserted_rows and inserted_rows > 0:
            logger.info(f"Inserted {inserted_rows} item(s).")

            # Test 3: Select data (fetch_one)
            # Assuming item with name 'MySQL Test Item 1' exists, or adapt query
            select_query_one = "SELECT * FROM mysql_test_items WHERE name = %s;"
            item = mysql_connector.execute_query(select_query_one, ("MySQL Test Item 1",), fetch_one=True)
            if item:
                logger.info(f"Selected item (fetch_one): {item}")

            # Test 4: Select all data (fetch_all)
            select_all_query = "SELECT * FROM mysql_test_items;"
            all_items = mysql_connector.execute_query(select_all_query, fetch_all=True)
            if all_items is not None: # Could be an empty list if table is empty or query fails
                logger.info(f"All items (fetch_all): {all_items}")

        # Test 5: Clean up (optional)
        # drop_table_query = "DROP TABLE IF EXISTS mysql_test_items;"
        # if mysql_connector.execute_query(drop_table_query, commit=True) is not None:
        #     logger.info("Table 'mysql_test_items' dropped.")
    else:
        logger.error("Failed to create 'mysql_test_items' table.")

    mysql_connector.close_pool()
