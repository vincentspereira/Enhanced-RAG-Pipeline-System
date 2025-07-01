import psycopg2
from psycopg2 import pool
import logging
from typing import Optional, List, Tuple, Dict, Any

# Assuming PostgresConfig is defined in config.manager, but to avoid direct import issues here for now:
# from ..config.manager import PostgresConfig # Use this if structure allows
# For now, we'll assume config is passed as a dictionary or an object with necessary attributes.

logger = logging.getLogger(__name__)

class PostgresConnector:
    def __init__(self, config: Any): # config should be compatible with PostgresConfig
        self.db_config = {
            "host": getattr(config, 'host', 'localhost'),
            "port": getattr(config, 'port', 5432),
            "user": getattr(config, 'username', 'postgres'),
            "password": getattr(config, 'password', 'postgres'), # Should come from secrets
            "dbname": getattr(config, 'database', 'rag_db'),
            "connect_timeout": getattr(config, 'connection_timeout', 10)
        }
        self.connection_pool = None
        try:
            # Initialize a connection pool (minconn=1, maxconn=5 example)
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                1,
                5,
                **self.db_config
            )
            logger.info(f"PostgreSQL connection pool initialized for {self.db_config['dbname']} on {self.db_config['host']}:{self.db_config['port']}")
            self.ping() # Initial ping to check connectivity
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL connection pool: {e}", exc_info=True)
            self.connection_pool = None # Ensure pool is None if init fails

    def get_connection(self):
        if not self.connection_pool:
            logger.error("PostgreSQL connection pool is not initialized.")
            raise ConnectionError("PostgreSQL connection pool not available.")
        try:
            return self.connection_pool.getconn()
        except Exception as e:
            logger.error(f"Failed to get connection from pool: {e}", exc_info=True)
            raise ConnectionError(f"Failed to get PostgreSQL connection: {e}")

    def release_connection(self, conn):
        if self.connection_pool and conn:
            self.connection_pool.putconn(conn)

    def ping(self) -> bool:
        """Checks if the database connection is alive."""
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            logger.info("PostgreSQL connection ping successful.")
            return True
        except Exception as e:
            logger.error(f"PostgreSQL connection ping failed: {e}", exc_info=True)
            return False
        finally:
            if conn:
                self.release_connection(conn)

    def execute_query(self, query: str, params: Optional[Union[List, Tuple, Dict]] = None, fetch_one: bool = False, fetch_all: bool = False, commit: bool = False) -> Optional[Any]:
        """
        Executes a SQL query.

        Args:
            query (str): The SQL query to execute.
            params (Optional[Union[List, Tuple, Dict]]): Parameters for the query.
            fetch_one (bool): If True, fetches one row.
            fetch_all (bool): If True, fetches all rows.
            commit (bool): If True, commits the transaction.

        Returns:
            Optional[Any]: Query result (single row, all rows, or None).
                           Returns rowcount for DML statements if not fetching.
        """
        conn = None
        result = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cur:
                cur.execute(query, params)

                if fetch_one:
                    result = cur.fetchone()
                elif fetch_all:
                    result = cur.fetchall()
                else:
                    try: # rowcount is not available for all statements (e.g. SELECT without fetch)
                        result = cur.rowcount
                    except psycopg2.ProgrammingError:
                        result = None # Or 0, or handle as needed

                if commit:
                    conn.commit()
            return result
        except Exception as e:
            if conn and not commit: # Rollback if commit wasn't intended or failed before commit
                try:
                    conn.rollback()
                except Exception as rb_err:
                    logger.error(f"Error during rollback: {rb_err}")
            logger.error(f"Error executing query '{query[:100]}...': {e}", exc_info=True)
            raise # Re-raise the exception to allow for specific handling by the caller
        finally:
            if conn:
                self.release_connection(conn)

    # --- Placeholder methods for common operations ---
    async def insert_data(self, table_name: str, data: Dict[str, Any]) -> Optional[int]:
        """Placeholder for inserting data."""
        # Example:
        # columns = ", ".join(data.keys())
        # placeholders = ", ".join(["%s"] * len(data))
        # query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        # return self.execute_query(query, list(data.values()), commit=True)
        logger.info(f"Placeholder: insert_data into {table_name} with {data}")
        return 1 # Simulate one row inserted

    async def get_data_by_id(self, table_name: str, item_id: Any, id_column: str = "id") -> Optional[Dict[str, Any]]:
        """Placeholder for fetching data by ID."""
        # Example:
        # query = f"SELECT * FROM {table_name} WHERE {id_column} = %s"
        # row = self.execute_query(query, (item_id,), fetch_one=True)
        # if row and cur.description: # Assuming cursor description is available
        #     return dict(zip([desc[0] for desc in cur.description], row))
        logger.info(f"Placeholder: get_data_by_id from {table_name} for ID {item_id}")
        return {"id": item_id, "data": "sample data"}

    def close_pool(self):
        """Closes all connections in the pool."""
        if self.connection_pool:
            try:
                self.connection_pool.closeall()
                logger.info("PostgreSQL connection pool closed.")
            except Exception as e:
                logger.error(f"Error closing PostgreSQL connection pool: {e}", exc_info=True)
            finally:
                self.connection_pool = None

# Example usage (for testing or direct use if not managed by a central app state)
if __name__ == '__main__':
    # This example requires a running PostgreSQL instance and proper credentials.
    # Replace with your actual config or load from a config file/env vars.
    class MockPostgresConfig:
        host = "localhost"
        port = 5432
        username = "your_user" # Replace
        password = "your_password" # Replace
        database = "your_db" # Replace
        connection_timeout = 5

    test_pg_config = MockPostgresConfig()

    pg_connector = None
    try:
        pg_connector = PostgresConnector(config=test_pg_config)
        if pg_connector.ping():
            print("Successfully connected to PostgreSQL.")

            # Example: Create a test table (idempotent)
            create_table_query = """
            CREATE TABLE IF NOT EXISTS test_items (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            pg_connector.execute_query(create_table_query, commit=True)
            print("Test table 'test_items' ensured.")

            # Example: Insert data
            insert_query = "INSERT INTO test_items (name) VALUES (%s) RETURNING id;"
            item_name = f"TestItem_{int(time.time())}"
            inserted_id = pg_connector.execute_query(insert_query, (item_name,), fetch_one=True, commit=True)
            if inserted_id:
                print(f"Inserted item with ID: {inserted_id[0]} and name: {item_name}")

                # Example: Fetch data
                select_query = "SELECT id, name, created_at FROM test_items WHERE id = %s;"
                row = pg_connector.execute_query(select_query, (inserted_id[0],), fetch_one=True)
                if row:
                    print(f"Fetched item: ID={row[0]}, Name={row[1]}, CreatedAt={row[2]}")

            # Example: Fetch all
            all_items = pg_connector.execute_query("SELECT name FROM test_items LIMIT 5;", fetch_all=True)
            print(f"Fetched up to 5 items: {all_items}")

    except ConnectionError as ce:
        print(f"Connection error: {ce}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if pg_connector:
            pg_connector.close_pool()
