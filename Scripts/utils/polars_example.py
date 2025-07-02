import polars as pl
import logging
import io # For creating an in-memory CSV

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_polars_examples():
    logger.info("--- Running Polars Examples ---")

    # --- Example 1: Creating a DataFrame from various sources ---
    logger.info("\n1. Creating DataFrames:")

    # From a dictionary
    data_dict = {
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "age": [25, 30, 35, 28, 32],
        "city": ["New York", "London", "Paris", "New York", "London"]
    }
    df_from_dict = pl.DataFrame(data_dict)
    logger.info("DataFrame from dictionary:\n" + str(df_from_dict))

    # From a list of rows (list of tuples or lists)
    data_rows = [
        (101, "Product A", 10.99, "Electronics"),
        (102, "Product B", 5.49, "Groceries"),
        (103, "Product C", 22.00, "Electronics"),
        (104, "Product D", 7.25, "Books"),
    ]
    df_from_rows = pl.DataFrame(data_rows, schema=["product_id", "product_name", "price", "category"])
    logger.info("\nDataFrame from list of rows:\n" + str(df_from_rows))

    # From an in-memory CSV string
    csv_data_string = """item_id,item_name,quantity,store_id
    A1,Apple,100,S1
    B1,Banana,150,S2
    A1,Apple,50,S1
    C1,Cherry,75,S2
    B1,Banana,200,S1
    """
    # Use io.StringIO to simulate a file
    csv_file_like = io.StringIO(csv_data_string)
    df_from_csv = pl.read_csv(csv_file_like)
    logger.info("\nDataFrame from CSV string:\n" + str(df_from_csv))

    # --- Example 2: Basic DataFrame operations ---
    logger.info("\n2. Basic DataFrame Operations:")
    logger.info(f"Shape of df_from_dict: {df_from_dict.shape}")
    logger.info(f"Schema of df_from_dict: {df_from_dict.schema}")
    logger.info(f"First 3 rows of df_from_dict (head):\n{df_from_dict.head(3)}")
    logger.info(f"Summary statistics of df_from_dict (describe):\n{df_from_dict.describe()}")

    # --- Example 3: Selecting columns ---
    logger.info("\n3. Selecting Columns:")
    selected_cols_df = df_from_dict.select(["name", "city"])
    logger.info("Selected 'name' and 'city' columns:\n" + str(selected_cols_df))

    # Selecting with expressions (e.g., renaming)
    selected_expr_df = df_from_dict.select(
        pl.col("name").alias("employee_name"),
        pl.col("age")
    )
    logger.info("\nSelected with expressions (renaming 'name'):\n" + str(selected_expr_df))

    # --- Example 4: Filtering rows ---
    logger.info("\n4. Filtering Rows:")
    filtered_df_age = df_from_dict.filter(pl.col("age") > 30)
    logger.info("Filtered where age > 30:\n" + str(filtered_df_age))

    filtered_df_city = df_from_dict.filter(pl.col("city") == "New York")
    logger.info("\nFiltered where city is 'New York':\n" + str(filtered_df_city))

    # Multiple conditions (AND)
    filtered_multi_cond = df_from_dict.filter(
        (pl.col("age") < 30) & (pl.col("city") == "New York")
    )
    logger.info("\nFiltered where age < 30 AND city is 'New York':\n" + str(filtered_multi_cond))

    # --- Example 5: Adding/Modifying columns (with_columns) ---
    logger.info("\n5. Adding/Modifying Columns:")
    df_with_new_col = df_from_dict.with_columns(
        (pl.col("age") * 2).alias("age_doubled"),
        pl.lit("USA").alias("country") # Add a literal column
    )
    logger.info("DataFrame with 'age_doubled' and 'country' columns:\n" + str(df_with_new_col))

    # --- Example 6: Grouping and Aggregation ---
    logger.info("\n6. Grouping and Aggregation (using df_from_csv):")
    logger.info("Original df_from_csv:\n" + str(df_from_csv))

    agg_df = df_from_csv.group_by("item_name").agg(
        pl.sum("quantity").alias("total_quantity"),
        pl.count().alias("num_entries"),
        pl.mean("quantity").alias("avg_quantity_per_entry")
    )
    logger.info("\nAggregated by 'item_name':\n" + str(agg_df))

    agg_by_store_item = df_from_csv.group_by(["store_id", "item_name"]).agg(
        pl.sum("quantity").alias("total_quantity_in_store")
    ).sort(["store_id", "item_name"]) # Sort for consistent output
    logger.info("\nAggregated by 'store_id' and 'item_name':\n" + str(agg_by_store_item))

    # --- Example 7: Sorting ---
    logger.info("\n7. Sorting:")
    sorted_df = df_from_dict.sort("age", descending=True)
    logger.info("Sorted by age (descending):\n" + str(sorted_df))

    # --- Example 8: Joins ---
    logger.info("\n8. Joins:")
    # Create another DataFrame for joining
    store_details_data = {
        "store_id": ["S1", "S2", "S3"],
        "store_location": ["Downtown", "Uptown", "Midtown"]
    }
    df_stores = pl.DataFrame(store_details_data)
    logger.info("Store details DataFrame:\n" + str(df_stores))

    joined_df = df_from_csv.join(df_stores, on="store_id", how="inner")
    logger.info("\nInner join of CSV data with store details:\n" + str(joined_df))

    left_joined_df = df_from_csv.join(df_stores, on="store_id", how="left")
    logger.info("\nLeft join of CSV data with store details:\n" + str(left_joined_df))


    # --- Example 9: Converting to other formats ---
    logger.info("\n9. Converting to other formats:")
    numpy_array = df_from_dict.to_numpy()
    logger.info(f"Converted to NumPy array (first 2 rows): \n{numpy_array[:2]}")

    # To list of dictionaries
    list_of_dicts = df_from_dict.to_dicts()
    logger.info(f"Converted to list of dictionaries (first 2 dicts): \n{list_of_dicts[:2]}")

    # To Pandas DataFrame (if pandas is installed)
    try:
        pandas_df = df_from_dict.to_pandas()
        logger.info(f"Converted to Pandas DataFrame (first 2 rows): \n{pandas_df.head(2)}")
    except ImportError:
        logger.warning("Pandas is not installed, skipping to_pandas() example.")
    except Exception as e:
        logger.warning(f"Could not convert to Pandas: {e}")


    logger.info("\n--- Polars Examples Finished ---")

if __name__ == "__main__":
    run_polars_examples()

    # To run this script:
    # Ensure polars is installed: pip install polars
    # Then execute: python Scripts/utils/polars_example.py
    #
    # This script demonstrates:
    # - DataFrame creation from various sources.
    # - Basic inspection (shape, schema, head, describe).
    # - Column selection and aliasing.
    # - Row filtering with single and multiple conditions.
    # - Adding and modifying columns using `with_columns`.
    # - Grouping data and performing aggregations (sum, count, mean).
    # - Sorting DataFrames.
    # - Performing joins (inner, left) between DataFrames.
    # - Converting Polars DataFrames to NumPy arrays, list of dictionaries, and (optionally) Pandas DataFrames.
