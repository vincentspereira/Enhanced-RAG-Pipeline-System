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
