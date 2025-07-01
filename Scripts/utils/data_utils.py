import polars as pl
from typing import Optional, Dict, List
import io
import logging

logger = logging.getLogger(__name__)

def example_polars_read_csv_from_string(csv_string_data: str) -> Optional[pl.DataFrame]:
    """
    Demonstrates reading CSV data from a string into a Polars DataFrame.

    Args:
        csv_string_data (str): A string containing CSV formatted data.
                               Example: "col_a,col_b\\n1,hello\\n2,world"

    Returns:
        Optional[pl.DataFrame]: A Polars DataFrame if successful, None otherwise.
    """
    try:
        # Polars can read directly from a file-like object.
        # For a string, we can use io.StringIO or io.BytesIO if it were bytes.
        # If your CSV string might have unicode issues, using BytesIO with utf-8 encoding is safer.
        csv_file_like = io.StringIO(csv_string_data)

        df = pl.read_csv(csv_file_like)
        logger.info(f"Successfully read CSV string into Polars DataFrame. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to read CSV string with Polars: {e}", exc_info=True)
        return None

def example_polars_transform_dataframe(df: pl.DataFrame, new_column_name: str = "col_c", value_to_add: int = 10) -> Optional[pl.DataFrame]:
    """
    Demonstrates a simple transformation on a Polars DataFrame:
    - Adds a new column by performing an operation on an existing numeric column (assuming 'col_a' exists and is numeric).
    - Filters rows based on a condition on 'col_a'.

    Args:
        df (pl.DataFrame): An input Polars DataFrame. Expected to have a numeric column 'col_a'.
        new_column_name (str): Name for the new column.
        value_to_add (int): Value to add to 'col_a' to create the new column.

    Returns:
        Optional[pl.DataFrame]: The transformed Polars DataFrame, or None if an error occurs.
    """
    if not isinstance(df, pl.DataFrame):
        logger.error("Input is not a Polars DataFrame.")
        return None

    try:
        if "col_a" not in df.columns:
            logger.error("Input DataFrame must contain a column named 'col_a'.")
            return None

        # Ensure 'col_a' is numeric or can be cast to numeric for the operation
        # This is a simple example; more robust type checking/casting might be needed.

        transformed_df = df.with_columns(
            (pl.col("col_a") + value_to_add).alias(new_column_name)
        ).filter(
            pl.col("col_a") > 1 # Example filter condition
        )
        logger.info(f"Polars DataFrame transformed. New shape: {transformed_df.shape}")
        return transformed_df
    except Exception as e:
        logger.error(f"Failed to transform Polars DataFrame: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Test example_polars_read_csv_from_string
    sample_csv_data = "col_a,col_b\n1,alpha\n2,beta\n3,gamma\n0,delta"
    print(f"\n--- Testing Polars CSV Read from String ---")
    print(f"Input CSV string:\n{sample_csv_data}")
    df_from_string = example_polars_read_csv_from_string(sample_csv_data)
    if df_from_string is not None:
        print("Polars DataFrame from string:")
        print(df_from_string)

        # Test example_polars_transform_dataframe
        print(f"\n--- Testing Polars DataFrame Transformation ---")
        transformed_df = example_polars_transform_dataframe(df_from_string, new_column_name="col_a_plus_10", value_to_add=10)
        if transformed_df is not None:
            print("Transformed Polars DataFrame (col_a > 1, col_a_plus_10 added):")
            print(transformed_df)
        else:
            print("Transformation failed.")
    else:
        print("Reading CSV from string failed.")

    # Example with a potentially problematic 'col_a' for transformation
    # (e.g., if 'col_a' was string type and couldn't be added to an int)
    sample_csv_data_str_col = "col_a,col_b\napple,alpha\nbanana,beta"
    print(f"\n--- Testing Polars Transformation with potentially non-numeric 'col_a' ---")
    df_str_col = example_polars_read_csv_from_string(sample_csv_data_str_col)
    if df_str_col is not None:
        print("DataFrame with string 'col_a':")
        print(df_str_col)
        transformed_df_str = example_polars_transform_dataframe(df_str_col) # This should log an error or fail gracefully
        if transformed_df_str is None:
            print("Transformation correctly handled error or returned None for non-numeric 'col_a' operation.")
        else:
            print("Transformed (unexpectedly?):")
            print(transformed_df_str)

    # --- DuckDB Examples ---
    print("\n--- Testing DuckDB Examples ---")
    import duckdb
    from pathlib import Path

    # Example 1: In-memory DB, query Pandas DataFrame
    print("\n1. Querying Pandas DataFrame with DuckDB (in-memory):")
    try:
        import pandas as pd
        pandas_df = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'category': ['A', 'B', 'A', 'C'],
            'value': [10.1, 20.2, 15.5, 5.0]
        })
        print("Original Pandas DataFrame:")
        print(pandas_df)

        # DuckDB can directly query pandas DataFrames registered as tables
        con = duckdb.connect(database=':memory:', read_only=False)
        # con.register('my_pandas_table', pandas_df) # No longer needed with modern DuckDB, can query directly

        result_df_pandas = con.execute("SELECT category, SUM(value) AS total_value FROM pandas_df GROUP BY category ORDER BY category").fetchdf()
        # Or using Polars for output: result_pl_pandas = con.execute("...").pl()
        con.close()

        print("Aggregated result from Pandas DataFrame via DuckDB:")
        print(result_df_pandas)

    except ImportError:
        print("Pandas not installed, skipping Pandas-DuckDB example.")
    except Exception as e:
        print(f"Error in Pandas-DuckDB example: {e}")

    # Example 2: Query Polars DataFrame
    print("\n2. Querying Polars DataFrame with DuckDB (in-memory):")
    try:
        # Assuming Polars DataFrame df_from_string exists from previous Polars example
        if df_from_string is not None and 'col_a' in df_from_string.columns and 'col_b' in df_from_string.columns:
            print("Original Polars DataFrame (df_from_string):")
            print(df_from_string)

            con = duckdb.connect(database=':memory:', read_only=False)
            # DuckDB can query Polars DataFrames directly if they are in scope
            # con.register('my_polars_table', df_from_string) # Not needed

            # Ensure col_a is numeric for sum, might need casting if it was read as string
            # For this example, assuming df_from_string['col_a'] is numeric or castable by duckdb
            try:
                result_pl_polars = con.execute("SELECT col_b, SUM(col_a) AS total_col_a FROM df_from_string GROUP BY col_b ORDER BY col_b").pl()
                print("Aggregated result from Polars DataFrame via DuckDB:")
                print(result_pl_polars)
            except Exception as e_query: # Catch potential DuckDB query errors (e.g. type mismatch)
                 print(f"DuckDB query error on Polars DF: {e_query}. 'col_a' might not be numeric.")
            finally:
                con.close()
        else:
            print("Polars DataFrame 'df_from_string' not available or suitable for DuckDB example.")

    except Exception as e:
        print(f"Error in Polars-DuckDB example: {e}")

    # Example 3: Directly query a CSV file
    print("\n3. Directly querying a CSV file with DuckDB:")
    temp_csv_path = Path("temp_duckdb_data.csv")
    with open(temp_csv_path, "w") as f:
        f.write("product_id,product_name,price,stock\n")
        f.write("101,Apple,1.50,100\n")
        f.write("102,Banana,0.75,150\n")
        f.write("103,Orange,1.25,80\n")
        f.write("101,Apple,1.55,50\n") # Duplicate product for aggregation

    try:
        con = duckdb.connect(database=':memory:', read_only=False)
        # DuckDB can query CSV files directly using read_csv_auto function or by path
        result_csv_query = con.execute(f"SELECT product_name, SUM(stock*price) AS total_value, AVG(price) as avg_price FROM read_csv_auto('{str(temp_csv_path)}') GROUP BY product_name ORDER BY product_name").pl()
        # Or: result_csv_query = con.execute(f"SELECT ... FROM '{str(temp_csv_path)}' ...").pl() # Simpler syntax often works
        con.close()

        print("Result from querying CSV directly via DuckDB:")
        print(result_csv_query)

    except Exception as e:
        print(f"Error in DuckDB CSV query example: {e}")
    finally:
        if temp_csv_path.exists():
            temp_csv_path.unlink()
            print(f"Cleaned up temporary CSV: {temp_csv_path}")
