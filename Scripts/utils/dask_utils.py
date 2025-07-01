import dask.dataframe as dd
import pandas as pd
import dask.array as da
import numpy as np
import logging
from typing import Optional
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

def example_dask_dataframe_from_pandas(pandas_df: pd.DataFrame, npartitions: Optional[int] = None) -> Optional[dd.DataFrame]:
    """
    Demonstrates creating a Dask DataFrame from a Pandas DataFrame.

    Args:
        pandas_df (pd.DataFrame): The input Pandas DataFrame.
        npartitions (Optional[int]): The number of partitions for the Dask DataFrame.
                                     If None, Dask will choose a default.
    Returns:
        Optional[dd.DataFrame]: The created Dask DataFrame, or None on error.
    """
    if not isinstance(pandas_df, pd.DataFrame):
        logger.error("Input is not a Pandas DataFrame.")
        return None

    npartitions = npartitions or 2 # Default to 2 partitions for small examples
    try:
        ddf = dd.from_pandas(pandas_df, npartitions=npartitions)
        logger.info(f"Successfully created Dask DataFrame from Pandas DataFrame with {ddf.npartitions} partitions.")
        return ddf
    except Exception as e:
        logger.error(f"Failed to create Dask DataFrame from Pandas: {e}", exc_info=True)
        return None

def example_dask_dataframe_from_csv(csv_file_path: str, blocksize: Optional[str] = "64MB") -> Optional[dd.DataFrame]:
    """
    Demonstrates creating a Dask DataFrame by reading a CSV file.
    Dask is particularly useful for CSVs larger than memory.

    Args:
        csv_file_path (str): Path to the CSV file.
        blocksize (Optional[str]): Size of blocks to chunk the CSV file into (e.g., "64MB", "128MB").
                                   Dask creates one partition per block.

    Returns:
        Optional[dd.DataFrame]: The created Dask DataFrame, or None on error.
    """
    try:
        # For this example, we assume the CSV has headers.
        # Dask can infer dtypes, or you can provide them for robustness.
        ddf = dd.read_csv(csv_file_path, blocksize=blocksize)
        logger.info(f"Successfully created Dask DataFrame from CSV '{csv_file_path}' with {ddf.npartitions} partitions.")
        return ddf
    except FileNotFoundError:
        logger.error(f"CSV file not found for Dask: {csv_file_path}")
        return None
    except Exception as e:
        logger.error(f"Failed to create Dask DataFrame from CSV '{csv_file_path}': {e}", exc_info=True)
        return None

def example_dask_computation(ddf: dd.DataFrame, column_name: str) -> Optional[Any]:
    """
    Demonstrates a simple computation (mean) on a Dask DataFrame column.
    The computation is only triggered when .compute() is called.

    Args:
        ddf (dd.DataFrame): The input Dask DataFrame.
        column_name (str): The name of the column to compute the mean of (must be numeric).

    Returns:
        Optional[Any]: The result of the computation, or None on error.
    """
    if not isinstance(ddf, dd.DataFrame):
        logger.error("Input is not a Dask DataFrame.")
        return None
    if column_name not in ddf.columns:
        logger.error(f"Column '{column_name}' not found in Dask DataFrame.")
        return None

    try:
        # Example: Calculate the mean of a column
        # This builds the task graph. Computation happens on .compute().
        mean_value = ddf[column_name].mean()

        logger.info(f"Dask task graph for mean of '{column_name}' created. Computing...")
        result = mean_value.compute() # Trigger computation
        logger.info(f"Computed mean of '{column_name}': {result}")
        return result
    except Exception as e:
        logger.error(f"Failed to perform Dask computation on column '{column_name}': {e}", exc_info=True)
        return None

def example_dask_array_computation(array_shape: Tuple[int, int] = (10000, 1000), chunks: Tuple[int, int] = (1000, 1000)) -> Optional[Any]:
    """
    Demonstrates a simple computation on a Dask Array.

    Args:
        array_shape (Tuple[int, int]): Shape of the Dask array to create.
        chunks (Tuple[int, int]): Chunk size for the Dask array.

    Returns:
        Optional[Any]: The result of the computation, or None on error.
    """
    try:
        # Create a large random Dask array
        x = da.random.random(array_shape, chunks=chunks)
        logger.info(f"Created Dask array with shape {x.shape} and chunks {x.chunks}")

        # Perform a computation (e.g., sum along an axis)
        # This builds the task graph.
        y = x.mean(axis=0)

        logger.info("Dask task graph for array mean created. Computing...")
        result = y.compute() # Trigger computation
        logger.info(f"Computed Dask array operation. Result shape: {result.shape}")
        return result
    except Exception as e:
        logger.error(f"Failed to perform Dask array computation: {e}", exc_info=True)
        return None


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("--- Dask DataFrame Examples ---")

    # Example 1: Dask DataFrame from Pandas
    print("\n1. Dask DataFrame from Pandas DataFrame:")
    sample_pandas_df = pd.DataFrame({
        'id': np.arange(100),
        'value': np.random.rand(100) * 100,
        'category': np.random.choice(['A', 'B', 'C'], size=100)
    })
    dask_df_from_pd = example_dask_dataframe_from_pandas(sample_pandas_df, npartitions=4)
    if dask_df_from_pd is not None:
        print(f"  Dask DF created. Number of partitions: {dask_df_from_pd.npartitions}")
        # Perform a computation
        mean_value_pd = example_dask_computation(dask_df_from_pd, 'value')
        if mean_value_pd is not None:
            print(f"  Computed mean of 'value' column: {mean_value_pd:.2f}")
            # For comparison with Pandas direct computation:
            # print(f"  Pandas direct mean: {sample_pandas_df['value'].mean():.2f}")

        # Example: Groupby and aggregation
        print("\n  Dask DataFrame Groupby Aggregation:")
        try:
            agg_result = dask_df_from_pd.groupby('category')['value'].mean().compute()
            print("  Mean 'value' per 'category':")
            print(agg_result)
        except Exception as e:
            print(f"  Error during Dask groupby: {e}")


    # Example 2: Dask DataFrame from CSV
    print("\n2. Dask DataFrame from CSV:")
    # Create a temporary CSV file for the example
    temp_dir = tempfile.mkdtemp()
    csv_path = Path(temp_dir) / "dask_sample_data.csv"

    # Create a slightly larger sample CSV to make blocksize more relevant
    num_rows_csv = 100000
    large_sample_df_for_csv = pd.DataFrame({
        'id': np.arange(num_rows_csv),
        'measurement': np.random.normal(loc=50, scale=10, size=num_rows_csv),
        'sensor': np.random.randint(1, 5, size=num_rows_csv)
    })
    large_sample_df_for_csv.to_csv(csv_path, index=False)
    print(f"  Created temporary CSV file: {csv_path} with {num_rows_csv} rows.")

    dask_df_from_csv = example_dask_dataframe_from_csv(str(csv_path), blocksize="1MB") # Small blocksize for more partitions
    if dask_df_from_csv is not None:
        print(f"  Dask DF from CSV created. Number of partitions: {dask_df_from_csv.npartitions}")
        print(f"  Columns: {dask_df_from_csv.columns}")
        # Perform a computation
        mean_measurement = example_dask_computation(dask_df_from_csv, 'measurement')
        if mean_measurement is not None:
            print(f"  Computed mean of 'measurement' column: {mean_measurement:.2f}")

    # Clean up temporary CSV
    try:
        os.remove(csv_path)
        os.rmdir(temp_dir)
        print(f"  Cleaned up temporary CSV and directory: {csv_path}")
    except Exception as e:
        logger.error(f"Error cleaning up temp Dask CSV: {e}")

    print("\n--- Dask Array Example ---")
    dask_array_result = example_dask_array_computation(array_shape=(5000, 2000), chunks=(1000, 500))
    if dask_array_result is not None:
        print(f"  Dask array computation example successful. Result starts with: {dask_array_result[:5]}")

    print("\nNote on Dask Schedulers:")
    print("The examples above use Dask's default local threaded scheduler.")
    print("For true parallelism across multiple cores or machines, you would configure Dask with:")
    print("- Local Schedulers: 'threads', 'processes' (from dask.distributed import Client; client = Client(processes=False) or Client(threads_per_worker=...))")
    print("- Distributed Scheduler: Set up a Dask cluster (dask-scheduler, dask-worker) and connect a Client to it.")
    print("  e.g., from dask.distributed import Client; client = Client('tcp://scheduler-address:8786')")

    print("\nDask Utils Example Done.")
