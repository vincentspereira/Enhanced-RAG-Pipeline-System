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
        # For RealDictCursor
        from psycopg2.extras import RealDictCursor

        try:
            # Initialize a connection pool (minconn=1, maxconn=5 example)
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                1,
                5,
                cursor_factory=RealDictCursor, # Use RealDictCursor for dict-like rows
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

    # --- pgVector Specific Examples ---
    # Note: Ensure the 'vector' extension is created in your PostgreSQL database:
    # CREATE EXTENSION IF NOT EXISTS vector;

    def create_table_with_vector_column(self, table_name: str, vector_dim: int, text_column_name: str = "text_content") -> None:
        """
        Example function to create a table with a text column and a pgVector vector column.
        This is an illustrative example; adapt schema as needed.
        """
        # Note: For HNSW or IVF indexes on vector columns for faster search,
        # you would add `USING hnsw (vector_column vector_l2_ops)` or similar to CREATE INDEX.
        # Example: CREATE INDEX ON items_with_vectors USING hnsw (embedding vector_l2_ops);
        query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            {text_column_name} TEXT,
            embedding VECTOR({vector_dim})
        );
        """
        try:
            self.execute_query(query, commit=True)
            logger.info(f"Table '{table_name}' with vector column (dim={vector_dim}) ensured.")
        except Exception as e:
            logger.error(f"Failed to create table '{table_name}' with vector column: {e}")
            # Depending on desired behavior, you might want to raise e here

    def insert_vector_data(self, table_name: str, text_content: str, vector: List[float]) -> Optional[int]:
        """
        Example function to insert text and its corresponding vector into a table.
        Returns the ID of the inserted row.
        """
        # pgVector expects vectors in the format '[f1,f2,f3...]' as a string
        vector_str = '[' + ','.join(map(str, vector)) + ']'
        query = f"INSERT INTO {table_name} (text_content, embedding) VALUES (%s, %s) RETURNING id;"
        try:
            result = self.execute_query(query, (text_content, vector_str), fetch_one=True, commit=True)
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Failed to insert vector data into '{table_name}': {e}")
            return None

    def search_similar_vectors(
        self,
        table_name: str,
        query_vector: List[float],
        top_k: int = 5,
        distance_metric: str = "cosine" # "cosine", "l2", "inner_product"
    ) -> List[Dict[str, Any]]:
        """
        Example function to search for similar vectors using pgVector distance operators.

        Args:
            table_name (str): Name of the table containing vectors.
            query_vector (List[float]): The vector to search against.
            top_k (int): Number of similar items to return.
            distance_metric (str): 'cosine', 'l2', or 'inner_product'.

        Returns:
            List of results, each a dict with id, text_content, and distance/similarity.
        """
        vector_str = '[' + ','.join(map(str, query_vector)) + ']'

        if distance_metric == "cosine":
            # <=> operator for cosine distance (0 is perfect similarity, 1 is opposite)
            # Score will be 1 - distance for similarity
            query = f"""
            SELECT id, text_content, embedding <=> %s AS distance
            FROM {table_name}
            ORDER BY embedding <=> %s
            LIMIT %s;
            """
            params = (vector_str, vector_str, top_k)
        elif distance_metric == "l2":
            # <-> operator for L2 distance (Euclidean)
            # Score could be negative distance or 1/(1+distance)
            query = f"""
            SELECT id, text_content, embedding <-> %s AS distance
            FROM {table_name}
            ORDER BY embedding <-> %s
            LIMIT %s;
            """
            params = (vector_str, vector_str, top_k)
        elif distance_metric == "inner_product":
            # <#> operator for inner product. For normalized vectors, this is cosine similarity * -1
            # (pgVector returns negative inner product for similarity search to use ASC order)
            # Score will be -distance (which is the inner product)
            query = f"""
            SELECT id, text_content, embedding <#> %s AS negative_inner_product
            FROM {table_name}
            ORDER BY embedding <#> %s
            LIMIT %s;
            """
            params = (vector_str, vector_str, top_k)
        else:
            logger.error(f"Unsupported distance metric for pgVector search: {distance_metric}")
            return []

        try:
            raw_results = self.execute_query(query, params, fetch_all=True)
            results = []
            if raw_results:
                for row_tuple in raw_results: # Assuming execute_query returns list of tuples if not DictCursor
                    # If using DictCursor from a modified execute_query, this would be simpler:
                    # row = dict(row_tuple) # If execute_query doesn't return dicts directly
                    row = row_tuple # Assuming execute_query now returns dicts due to pool change

                    # Convert distance to similarity score (higher is better)
                    if distance_metric == "cosine":
                        # distance is 0 (identical) to 2 (opposite). 1-distance gives similarity.
                        score = 1.0 - float(row['distance'])
                    elif distance_metric == "l2":
                        # L2 distance is >= 0. Smaller is better. Convert to a similarity score.
                        score = 1.0 / (1.0 + float(row['distance']))
                    elif distance_metric == "inner_product":
                        # pgVector's <#> returns negative inner product for ORDER BY ASC.
                        # So, score = -negative_inner_product to get actual inner product (similarity).
                        score = -float(row['negative_inner_product'])
                    else:
                        score = 0.0 # Should not happen if metric is validated

                    results.append({
                        "id": row['id'],
                        "text_content": row['text_content'],
                        "score": score,
                        "original_distance_metric": distance_metric,
                        "original_distance_value": float(row.get('distance', row.get('negative_inner_product', 0)))
                    })
            return results
        except Exception as e:
            logger.error(f"Failed to search similar vectors in '{table_name}' using pgVector: {e}")
            return []

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
