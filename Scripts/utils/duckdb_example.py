import duckdb
import pandas as pd # Pandas is often used with DuckDB, though not strictly required
import polars as pl # Polars can also be used
import logging
import os
import io

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dummy CSV data for examples
DUMMY_CSV_DATA = """product_id,product_name,category,price,stock_quantity
101,Laptop Pro,Electronics,1200.00,50
102,Wireless Mouse,Electronics,25.50,150
103,Organic Apples,Groceries,3.99,200
104,Coffee Beans,Groceries,15.75,80
105,The Great Novel,Books,12.99,120
106,Gaming Keyboard,Electronics,75.00,75
107, नोटबुक,स्टेशनरी,1.50,300
""" # Added a row with non-ASCII characters for encoding test

# Ensure a directory for saving a persistent DuckDB database exists
DB_DIR = "data/duckdb_databases"
os.makedirs(DB_DIR, exist_ok=True)
PERSISTENT_DB_FILE_PATH = os.path.join(DB_DIR, "my_persistent_duck.db")


def run_duckdb_examples():
    logger.info("--- Running DuckDB Examples ---")

    # --- Example 1: In-memory database and querying a Pandas DataFrame ---
    logger.info("\n1. In-memory DB: Querying a Pandas DataFrame")

    # Create a Pandas DataFrame
    data_pandas = {
        'id': [1, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Charlie', 'David'],
        'salary': [70000, 80000, 65000, 90000]
    }
    pandas_df = pd.DataFrame(data_pandas)
    logger.info("Original Pandas DataFrame:\n" + str(pandas_df))

    # Connect to an in-memory DuckDB database
    # Each connection creates a new in-memory database unless a path is specified.
    conn_memory = duckdb.connect(database=':memory:', read_only=False)

    # Register the Pandas DataFrame as a DuckDB table (zero-copy, DuckDB queries it directly)
    # conn_memory.register('employees_pandas_table', pandas_df) # Old method
    # DuckDB now prefers to query DataFrames directly or use CREATE TABLE AS SELECT

    # Query the Pandas DataFrame directly (DuckDB is smart enough to handle this)
    try:
        result_pandas_direct = conn_memory.execute("SELECT name, salary FROM pandas_df WHERE salary > 70000 ORDER BY salary DESC").fetchall()
        logger.info("Query result (direct query on Pandas DF where salary > 70000):\n" + str(result_pandas_direct))
    except Exception as e:
        logger.error(f"Error querying Pandas DF directly: {e}. Trying registration method.")
        # Fallback or alternative: register and query
        conn_memory.register('employees_pandas_table_reg', pandas_df)
        result_pandas_registered = conn_memory.execute("SELECT name, salary FROM employees_pandas_table_reg WHERE salary > 70000 ORDER BY salary DESC").fetchall()
        logger.info("Query result (registered Pandas DF where salary > 70000):\n" + str(result_pandas_registered))


    # --- Example 2: Querying a Polars DataFrame ---
    logger.info("\n2. In-memory DB: Querying a Polars DataFrame")
    data_polars = {
        'item': ['A', 'B', 'A', 'C', 'B'],
        'value': [10, 20, 15, 5, 25]
    }
    polars_df = pl.DataFrame(data_polars)
    logger.info("Original Polars DataFrame:\n" + str(polars_df))

    # Query the Polars DataFrame directly
    result_polars_direct = conn_memory.execute("SELECT item, SUM(value) as total_value FROM polars_df GROUP BY item ORDER BY item").pl() # Fetch as Polars DataFrame
    logger.info("Query result (direct query on Polars DF, sum of values by item):\n" + str(result_polars_direct))


    # --- Example 3: Reading and querying a CSV file ---
    logger.info("\n3. In-memory DB: Reading and Querying a CSV")

    # Create a dummy CSV file for this example
    dummy_csv_path = os.path.join(DB_DIR, "dummy_products.csv")
    with open(dummy_csv_path, "w", encoding="utf-8") as f: # Specify UTF-8 for wider character support
        f.write(DUMMY_CSV_DATA)

    # DuckDB can query CSV files directly
    # Use read_csv_auto to automatically detect schema and read the CSV
    result_csv = conn_memory.execute(f"""
        SELECT category, COUNT(*) as num_products, AVG(price) as avg_price
        FROM read_csv_auto('{dummy_csv_path}')
        GROUP BY category
        ORDER BY num_products DESC
    """).fetchdf() # Fetch as Pandas DataFrame
    logger.info(f"Query result (from CSV file '{dummy_csv_path}', aggregated by category):\n" + str(result_csv))


    # --- Example 4: Persistent database ---
    logger.info(f"\n4. Persistent DB: Storing and Querying from '{PERSISTENT_DB_FILE_PATH}'")

    # Clean up previous persistent DB if it exists for a clean test run
    if os.path.exists(PERSISTENT_DB_FILE_PATH):
        os.remove(PERSISTENT_DB_FILE_PATH)
        logger.info(f"Removed existing persistent DB file: {PERSISTENT_DB_FILE_PATH}")

    conn_persistent = duckdb.connect(database=PERSISTENT_DB_FILE_PATH, read_only=False)

    # Create a table and insert data from the CSV
    conn_persistent.execute(f"""
        CREATE TABLE IF NOT EXISTS products AS
        SELECT * FROM read_csv_auto('{dummy_csv_path}');
    """)
    logger.info(f"Created table 'products' in persistent DB from CSV '{dummy_csv_path}'.")

    # Query the persistent table
    result_persistent = conn_persistent.execute("""
        SELECT product_name, price
        FROM products
        WHERE category = 'Electronics' AND price > 50
        ORDER BY price
    """).fetchall()
    logger.info("Query result (Electronics products with price > 50 from persistent DB):\n" + str(result_persistent))

    # Close the persistent connection (important to flush changes to disk)
    conn_persistent.close()
    logger.info(f"Closed persistent DB connection. Data saved to {PERSISTENT_DB_FILE_PATH}")

    # Reconnect to the persistent DB to verify data persistence
    logger.info("\nReconnecting to persistent DB to verify data...")
    conn_reconnect = duckdb.connect(database=PERSISTENT_DB_FILE_PATH, read_only=True) # Open in read-only
    result_reconnect = conn_reconnect.execute("SELECT COUNT(*) FROM products").fetchone()
    logger.info(f"Number of rows in 'products' table after reconnect: {result_reconnect[0] if result_reconnect else 'Error'}")
    conn_reconnect.close()


    # --- Example 5: Exporting query results ---
    logger.info("\n5. Exporting Query Results")
    export_csv_path = os.path.join(DB_DIR, "exported_electronics.csv")
    export_parquet_path = os.path.join(DB_DIR, "exported_groceries.parquet")

    conn_memory.execute(f"""
        COPY (
            SELECT * FROM read_csv_auto('{dummy_csv_path}') WHERE category = 'Electronics'
        ) TO '{export_csv_path}' (HEADER, DELIMITER ',');
    """)
    logger.info(f"Exported 'Electronics' products to CSV: {export_csv_path}")

    conn_memory.execute(f"""
        COPY (
            SELECT * FROM read_csv_auto('{dummy_csv_path}') WHERE category = 'Groceries'
        ) TO '{export_parquet_path}' (FORMAT PARQUET);
    """)
    logger.info(f"Exported 'Groceries' products to Parquet: {export_parquet_path}")

    # Clean up dummy CSV file
    if os.path.exists(dummy_csv_path):
        os.remove(dummy_csv_path)

    # Close the in-memory connection
    conn_memory.close()
    logger.info("\n--- DuckDB Examples Finished ---")

if __name__ == "__main__":
    run_duckdb_examples()

    # To run this script:
    # Ensure duckdb and pandas/polars are installed:
    #   pip install duckdb pandas polars
    # Then execute: python Scripts/utils/duckdb_example.py
    #
    # This script demonstrates:
    # - Connecting to an in-memory DuckDB database.
    # - Directly querying Pandas and Polars DataFrames using SQL.
    # - Reading and querying data directly from CSV files.
    # - Creating and using a persistent file-based DuckDB database.
    # - Creating tables from CSV data.
    # - Exporting query results to CSV and Parquet files.
    #
    # Check the 'data/duckdb_databases/' directory for the persistent DB file
    # and exported files after running.
