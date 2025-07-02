import os
import psycopg2
from psycopg2 import pool
import logging

# Configure logging
# Ensure your main application setup configures logging. For a library/module,
# it's often better to just get a logger and let the application configure handlers.
# logging.basicConfig(level=logging.INFO) # Avoid basicConfig in libraries
logger = logging.getLogger(__name__)

# Attempt to import the conceptual config_loader, will need to ensure it's accessible
try:
    from Scripts.utils.config_loader import get_config_value
except ImportError:
    # This fallback is problematic if Scripts.utils is not in PYTHONPATH during execution.
    # It's better to ensure the environment is set up correctly.
    # For robustness in case of direct script execution where PYTHONPATH might not be set:
    import sys
    import os
    # Add Scripts directory to Python path if utils is not found directly
    # This assumes postgres_connector.py is in Scripts/storage/
    # and config_loader.py is in Scripts/utils/
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        from utils.config_loader import get_config_value
    except ImportError as e:
        logger.error(f"Failed to import get_config_value. Ensure Scripts/utils/config_loader.py exists and PYTHONPATH is set correctly. Error: {e}")
        # A simple os.getenv fallback for critical parameters if all else fails, though not ideal.
        def get_config_value(env_var_name, yaml_path=None, default=None):
            val = os.getenv(env_var_name, default)
            logger.warning(f"Critical fallback for get_config_value: returning '{val}' for '{env_var_name}'")
            return val

class PostgresConnector:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(PostgresConnector, cls).__new__(cls)
        return cls._instance

    def __init__(self, min_conn=1, max_conn=5):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.db_host = get_config_value("POSTGRES_HOST", yaml_path="database.postgres.host", default="localhost")
        self.db_port = int(get_config_value("POSTGRES_PORT", yaml_path="database.postgres.port", default=5432))
        self.db_user = get_config_value("POSTGRES_USER", yaml_path="database.postgres.user", default="postgres")
        self.db_password = get_config_value("POSTGRES_PASSWORD", yaml_path="database.postgres.password", default="password")
        self.db_name = get_config_value("POSTGRES_DB", yaml_path="database.postgres.dbname", default="mydatabase")

        self.min_conn = int(get_config_value("POSTGRES_MIN_CONN", default=min_conn))
        self.max_conn = int(get_config_value("POSTGRES_MAX_CONN", default=max_conn))

        self.pool = None
        self._initialized = False # Mark that init logic is about to run

        try:
            self.pool = psycopg2.pool.SimpleConnectionPool(
                self.min_conn,
                self.max_conn,
                host=self.db_host,
                port=self.db_port,
                user=self.db_user,
                password=self.db_password,
                dbname=self.db_name
            )
            if self.pool:
                logger.info(f"PostgreSQL connection pool created successfully for {self.db_user}@{self.db_host}:{self.db_port}/{self.db_name}")
                self._initialized = True # Mark as fully initialized
            else:
                logger.error("Failed to create PostgreSQL connection pool.")
        except (Exception, psycopg2.Error) as error:
            logger.error(f"Error while connecting to PostgreSQL or creating pool: {error}")
            self.pool = None # Ensure pool is None if creation failed

    def get_connection(self):
        if not self.pool or not self._initialized:
            logger.error("Connection pool is not initialized.")
            # Optionally, try to re-initialize the pool here if it makes sense for the application
            # self.__init__(self.min_conn, self.max_conn) # Be careful with recursion
            # if not self.pool or not self._initialized:
            #     return None
            return None
        try:
            return self.pool.getconn()
        except (Exception, psycopg2.Error) as error:
            logger.error(f"Error getting connection from pool: {error}")
            return None

    def release_connection(self, conn):
        if not self.pool or not self._initialized:
            logger.warning("Connection pool is not initialized. Cannot release connection.")
            return
        if conn:
            try:
                self.pool.putconn(conn)
            except (Exception, psycopg2.Error) as error:
                logger.error(f"Error releasing connection to pool: {error}")

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
                return True # Or cursor.rowcount or other relevant info

            if fetch_one:
                return cursor.fetchone()
            if fetch_all:
                return cursor.fetchall()

            # If not fetching and not committing (e.g. DDL without commit, or multi-statement transaction part)
            # Return True or rowcount to indicate success
            return True

        except (Exception, psycopg2.Error) as error:
            logger.error(f"Error executing query: {query} with params: {params}. Error: {error}")
            if conn and commit: # If error during commit, rollback might be needed
                try:
                    conn.rollback()
                    logger.info("Transaction rolled back due to error during commit.")
                except Exception as rb_error:
                    logger.error(f"Error during rollback: {rb_error}")
            return None
        finally:
            if cursor:
                cursor.close()
            if conn:
                self.release_connection(conn)

    def close_pool(self):
        if self.pool and self._initialized:
            try:
                self.pool.closeall()
                logger.info("PostgreSQL connection pool closed.")
                self._initialized = False # Mark as no longer initialized
            except (Exception, psycopg2.Error) as error:
                logger.error(f"Error closing PostgreSQL connection pool: {error}")
        self.pool = None


# Example Usage (for testing purposes, typically done by services using the connector)
if __name__ == '__main__':
    # Set environment variables for testing if not using a config file or actual env vars
    # os.environ["POSTGRES_HOST"] = "localhost"
    # os.environ["POSTGRES_USER"] = "youruser"
    # os.environ["POSTGRES_PASSWORD"] = "yourpassword"
    # os.environ["POSTGRES_DB"] = "yourdb"

    pg_connector = PostgresConnector()

    if not pg_connector.pool or not pg_connector._initialized:
        logger.error("PostgresConnector not initialized. Exiting test.")
        exit()

    # Test 1: Create a table (DDL)
    create_table_query = """
    CREATE TABLE IF NOT EXISTS test_items (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        quantity INTEGER
    );
    """
    # DDL statements often don't need explicit commit in psycopg2 default (autocommit mode for DDL might be active)
    # but it's good practice for multi-statement transactions or if you change session characteristics.
    # For simplicity, we assume default behavior or that commit=True handles it.
    if pg_connector.execute_query(create_table_query, commit=True): # Commit can be true for DDL
        logger.info("Table 'test_items' created or already exists.")

        # Test 2: Insert data (DML with commit)
        insert_query = "INSERT INTO test_items (name, quantity) VALUES (%s, %s) RETURNING id;"
        item_id = pg_connector.execute_query(insert_query, ("Test Item 1", 10), fetch_one=True, commit=True)
        if item_id:
            logger.info(f"Inserted item with ID: {item_id[0]}")

            # Test 3: Select data (DQL)
            select_query = "SELECT * FROM test_items WHERE id = %s;"
            item = pg_connector.execute_query(select_query, (item_id[0],), fetch_one=True)
            if item:
                logger.info(f"Selected item: {item}")

        # Test 4: Select all data
        select_all_query = "SELECT * FROM test_items;"
        all_items = pg_connector.execute_query(select_all_query, fetch_all=True)
        if all_items is not None: # Could be an empty list
             logger.info(f"All items: {all_items}")

        # Test 5: Clean up (optional)
        # drop_table_query = "DROP TABLE test_items;"
        # if pg_connector.execute_query(drop_table_query, commit=True):
        # logger.info("Table 'test_items' dropped.")

    else:
        logger.error("Failed to create 'test_items' table.")

    pg_connector.close_pool()

    # --- pgVector Example Notes ---
    # To use pgVector with this connector:
    # 1. Ensure the pgVector extension is installed in your PostgreSQL database.
    #    Connect to your DB (e.g., via psql) and run: CREATE EXTENSION IF NOT EXISTS vector;
    #
    # 2. Create a table with a vector column:
    #    CREATE TABLE items_with_vectors (
    #        id SERIAL PRIMARY KEY,
    #        name TEXT,
    #        embedding VECTOR(3) -- Example: 3 dimensions, adjust to your embedding model's output size
    #    );
    #
    # 3. Insert data with vector embeddings:
    #    (Assuming 'embedding_vector' is a Python list/tuple like [0.1, 0.2, 0.3])
    #    insert_vector_query = "INSERT INTO items_with_vectors (name, embedding) VALUES (%s, %s);"
    #    pg_connector.execute_query(insert_vector_query, ("My Item", embedding_vector), commit=True)
    #    Note: psycopg2 automatically adapts Python lists/tuples to PostgreSQL array syntax, which pgVector understands for input.
    #
    # 4. Perform similarity search:
    #    (Assuming 'query_vector' is a Python list/tuple for the query embedding)
    #    similarity_search_query = """
    #    SELECT name, embedding <-> %s AS distance
    #    FROM items_with_vectors
    #    ORDER BY embedding <-> %s
    #    LIMIT 5;
    #    """
    #    results = pg_connector.execute_query(similarity_search_query, (query_vector, query_vector), fetch_all=True)
    #    if results:
    #        for row in results:
    #            print(f"Name: {row['name']}, Distance: {row['distance']}") # Assuming DictCursor
    #
    # Indexing for performance (example using HNSW for cosine distance):
    #    CREATE INDEX ON items_with_vectors USING hnsw (embedding vector_cosine_ops);
    # Consult pgVector documentation for appropriate index types based on your vector size and distance metric.
