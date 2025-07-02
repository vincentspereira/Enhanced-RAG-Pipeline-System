import dask.dataframe as dd
import pandas as pd
import dask.array as da
import numpy as np
import logging
import os
import shutil # For cleaning up dummy data

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directory for dummy CSV files
DUMMY_DATA_DIR = "data/dask_dummy_csv_data"

def setup_dummy_csv_files(num_files=3, num_rows_per_file=10000):
    """Creates dummy CSV files for Dask DataFrame examples."""
    os.makedirs(DUMMY_DATA_DIR, exist_ok=True)
    logger.info(f"Creating {num_files} dummy CSV files in {DUMMY_DATA_DIR}...")
    for i in range(num_files):
        file_path = os.path.join(DUMMY_DATA_DIR, f"data_part_{i}.csv")
        df = pd.DataFrame({
            'id': range(i * num_rows_per_file, (i + 1) * num_rows_per_file),
            'value1': np.random.rand(num_rows_per_file) * 100,
            'value2': np.random.randint(0, 1000, size=num_rows_per_file),
            'category': np.random.choice(['A', 'B', 'C', 'D'], size=num_rows_per_file)
        })
        df.to_csv(file_path, index=False)
    logger.info("Dummy CSV files created.")

def cleanup_dummy_csv_files():
    """Removes the dummy CSV files and directory."""
    if os.path.exists(DUMMY_DATA_DIR):
        shutil.rmtree(DUMMY_DATA_DIR)
        logger.info(f"Cleaned up dummy CSV data directory: {DUMMY_DATA_DIR}")

def run_dask_dataframe_examples():
    logger.info("\n--- Running Dask DataFrame Examples ---")
    setup_dummy_csv_files()

    # --- Example 1: Reading multiple CSV files into a Dask DataFrame ---
    logger.info("\n1. Reading multiple CSV files into a Dask DataFrame:")
    # Dask reads CSVs lazily by default. Computation happens when you call .compute() or an action.
    # The number of partitions will often correspond to the number of files.
    ddf = dd.read_csv(os.path.join(DUMMY_DATA_DIR, "data_part_*.csv"))
    logger.info(f"Dask DataFrame created from CSVs. Number of partitions: {ddf.npartitions}")
    logger.info(f"First 5 rows (head operation, triggers computation for a small part):\n{ddf.head()}")

    # --- Example 2: Basic computations ---
    logger.info("\n2. Basic computations on Dask DataFrame:")
    # Calculate the mean of 'value1'. This is still a lazy operation.
    mean_value1_lazy = ddf['value1'].mean()
    logger.info(f"Mean of 'value1' (lazy Dask object): {mean_value1_lazy}")

    # To get the actual result, call .compute()
    mean_value1_computed = mean_value1_lazy.compute()
    logger.info(f"Mean of 'value1' (computed): {mean_value1_computed}")

    # --- Example 3: Filtering and then computing ---
    logger.info("\n3. Filtering and then computing:")
    filtered_ddf = ddf[ddf['category'] == 'A']
    count_category_a_lazy = len(filtered_ddf) # len() on Dask DataFrame is also lazy for row count

    # Compute the count
    count_category_a_computed = count_category_a_lazy.compute() # This can be less efficient than specific count operations
    # More efficient way to count after filter:
    # count_category_a_computed = filtered_ddf.shape[0].compute() or filtered_ddf['id'].count().compute()

    logger.info(f"Number of rows where category is 'A' (computed): {count_category_a_computed}")
    logger.info(f"First 3 rows of category 'A' (computed):\n{filtered_ddf.head(3)}")


    # --- Example 4: Groupby and aggregation ---
    logger.info("\n4. Groupby and aggregation:")
    # Group by 'category' and calculate sum of 'value2' and mean of 'value1'
    aggregated_ddf_lazy = ddf.groupby('category').agg({'value2': 'sum', 'value1': 'mean'})
    logger.info(f"Aggregated Dask DataFrame (lazy):\n{aggregated_ddf_lazy}")

    aggregated_ddf_computed = aggregated_ddf_lazy.compute()
    logger.info(f"Aggregated Dask DataFrame (computed):\n{aggregated_ddf_computed}")

    # --- Example 5: Using map_partitions ---
    logger.info("\n5. Using map_partitions to apply a function to each partition:")
    # Example: Add a new column based on existing ones, per partition
    def process_partition(pdf: pd.DataFrame) -> pd.DataFrame:
        # pdf is a Pandas DataFrame (one partition of the Dask DataFrame)
        pdf['value1_plus_value2'] = pdf['value1'] + pdf['value2']
        return pdf

    # Define the metadata (schema) for the output of map_partitions
    # Dask needs to know the dtypes of the resulting DataFrame.
    meta = ddf.dtypes.to_dict()
    meta['value1_plus_value2'] = 'float64' # Or infer more robustly

    ddf_with_new_col = ddf.map_partitions(process_partition, meta=meta)
    logger.info(f"Dask DataFrame with new column (lazy). Partitions: {ddf_with_new_col.npartitions}")
    logger.info(f"First 5 rows of Dask DataFrame with new column (computed):\n{ddf_with_new_col.head()}")

    # --- Example 6: Converting Dask DataFrame to Pandas DataFrame ---
    # logger.info("\n6. Converting Dask DataFrame to Pandas DataFrame (be cautious with large data):")
    # if ddf_with_new_col.npartitions * num_rows_per_file < 100000: # Small enough to convert
    #     pandas_from_dask = ddf_with_new_col.compute()
    #     logger.info(f"Converted to Pandas DataFrame. Shape: {pandas_from_dask.shape}")
    # else:
    #     logger.info("Skipping full conversion to Pandas as data might be too large for memory.")

    cleanup_dummy_csv_files()

def run_dask_array_examples():
    logger.info("\n--- Running Dask Array Examples ---")

    # --- Example 1: Creating Dask Arrays ---
    logger.info("\n1. Creating Dask Arrays:")
    # From NumPy array, chunked
    numpy_arr = np.arange(10000).reshape(100, 100)
    dask_arr_from_numpy = da.from_array(numpy_arr, chunks=(50, 50)) # 2x2 chunks of (50,50) blocks
    logger.info(f"Dask Array from NumPy array: chunks={dask_arr_from_numpy.chunks}, shape={dask_arr_from_numpy.shape}")

    # Create a large random Dask array
    large_dask_arr = da.random.random((10000, 1000), chunks=(1000, 1000)) # 10x1 chunks
    logger.info(f"Large random Dask Array: chunks={large_dask_arr.chunks}, shape={large_dask_arr.shape}")

    # --- Example 2: Basic computations on Dask Arrays ---
    logger.info("\n2. Basic computations on Dask Arrays:")
    sum_large_arr_lazy = large_dask_arr.sum()
    logger.info(f"Sum of large_dask_arr (lazy Dask object): {sum_large_arr_lazy}")

    sum_large_arr_computed = sum_large_arr_lazy.compute()
    logger.info(f"Sum of large_dask_arr (computed): {sum_large_arr_computed}")

    # Element-wise operations are also lazy
    result_array_lazy = (large_dask_arr * 2) + 1
    logger.info(f"Result of (large_dask_arr * 2) + 1 (lazy Dask object): {result_array_lazy}")

    # Compute a slice of it
    # result_slice_computed = result_array_lazy[:5, :5].compute()
    # logger.info(f"Slice of computed result array:\n{result_slice_computed}")


def conceptual_notes_for_dask():
    logger.info("\n--- Conceptual Notes on Using Dask in this RAG System ---")
    logger.info("""
    Dask can be beneficial in several areas of a RAG (Retrieval Augmented Generation) system,
    especially when dealing with large volumes of data or computationally intensive tasks:

    1.  **Parallelizing Document Ingestion & Preprocessing**:
        *   Reading many files (PDFs, TXTs, etc.) concurrently from storage (local, S3).
            `dask.bag` or `dask.delayed` could read files in parallel.
        *   Text extraction from these files can be parallelized using `dask.delayed` or by
            applying a function over partitions of a Dask Bag/DataFrame of file paths.
        *   Chunking large documents: A document could be a single item in a Dask Bag,
            and a chunking function applied to each.

    2.  **Parallelizing Embedding Generation**:
        *   If you have many text chunks to embed, this is an "embarrassingly parallel" task.
        *   A Dask DataFrame containing text chunks can have an embedding function applied
            to its partitions using `map_partitions`. Each worker would handle a subset
            of chunks.
        *   Care must be taken with GPU resources if using GPU-based embedding models. Dask
            workers would need to be configured to access GPUs correctly, possibly one GPU
            per worker or careful scheduling.

    3.  **Large-Scale Data Analysis (e.g., on user feedback or document metadata)**:
        *   If you collect extensive analytics or have large metadata stores, Dask DataFrames
            can perform distributed aggregations, filtering, and joins, similar to Spark
            but often with a lighter setup for single-machine parallelism or smaller clusters.

    4.  **Distributed Training/Fine-tuning (Advanced)**:
        *   Libraries like `dask-ml` can help distribute certain machine learning tasks,
            though fine-tuning large language models often requires specialized distributed
            training frameworks (like PyTorch Distributed, DeepSpeed, etc.). Dask could play
            a role in the data preparation stages for such training.

    **Considerations when using Dask**:
    *   **Overhead**: For very small tasks, Dask's overhead might outweigh benefits. It shines
        with larger-than-memory datasets or tasks that can be significantly parallelized.
    *   **Scheduler**: Dask offers different schedulers (threaded, multiprocessing, distributed).
        The distributed scheduler is powerful for multi-machine clusters but adds setup complexity.
        For single-machine parallelism, multiprocessing is often a good choice for CPU-bound tasks.
    *   **Data Transfer**: When using distributed Dask, be mindful of data transfer costs
        between workers.
    *   **Memory Management**: Chunk sizes and partition strategies are important to manage
        memory effectively on workers.
    *   **Integration with existing code**: `dask.delayed` can be a good way to parallelize
        existing Python code with minimal changes.

    Using Dask effectively often means rethinking parts of your data pipeline to expose parallelism.
    """)

if __name__ == "__main__":
    run_dask_dataframe_examples()
    run_dask_array_examples()
    conceptual_notes_for_dask()

    # To run this script:
    # Ensure dask and pandas are installed: pip install "dask[dataframe]" pandas numpy
    # Then execute: python Scripts/utils/dask_example.py
    #
    # This script demonstrates:
    # - Creating and operating on Dask DataFrames (reading CSVs, computations, filtering, groupby).
    # - Using map_partitions for custom per-partition logic.
    # - Creating and operating on Dask Arrays.
    # - Conceptual notes on how Dask could be applied in this RAG system.
    #
    # It will create and then clean up a 'data/dask_dummy_csv_data' directory.
