from google.cloud import bigquery
from google.cloud.exceptions import NotFound, GoogleCloudError
import logging
from typing import Optional, List, Tuple, Dict, Any, Union

# Assuming BigQueryConfig is defined in config.manager:
# from ..config.manager import BigQueryConfig

logger = logging.getLogger(__name__)

class BigQueryConnector:
    def __init__(self, config: Any): # config should be compatible with BigQueryConfig
        # project_id can be set via GOOGLE_CLOUD_PROJECT env var,
        # or passed explicitly in config.
        self.project_id = getattr(config, 'project_id', None)
        self.dataset_id = getattr(config, 'dataset_id', None) # Default dataset for operations
        # Location can also be part of the config if needed for queries or table creation
        # self.location = getattr(config, 'location', 'US')

        self.client: Optional[bigquery.Client] = None
        try:
            # If project_id is None, the client will try to infer it from the environment.
            self.client = bigquery.Client(project=self.project_id)
            logger.info(f"BigQuery client initialized for project '{self.client.project}'.")
            if self.ping(): # Initial ping
                 logger.info("Successfully connected to BigQuery.")
        except GoogleCloudError as e:
            logger.error(f"Failed to initialize BigQuery client: {e}", exc_info=True)
            self.client = None
            raise ConnectionError(f"BigQuery client initialization failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error initializing BigQuery client: {e}", exc_info=True)
            self.client = None
            raise

    def close_connection(self):
        """Closes the BigQuery client resources (if any explicit close is needed)."""
        # google-cloud-python clients typically manage connections automatically
        # and don't always have an explicit close() method for HTTP-based clients.
        # However, it's good practice if one exists or to clear the client reference.
        if self.client:
            # self.client.close() # Uncomment if client has a close method. As of recent versions, it might not.
            logger.info("BigQuery client resources released (if applicable).")
            self.client = None

    def __enter__(self):
        # Re-initialize client if it's None (e.g., after a previous close or init failure)
        if not self.client:
            try:
                self.client = bigquery.Client(project=self.project_id)
                logger.info(f"BigQuery client re-initialized for project '{self.client.project}'.")
            except Exception as e:
                logger.error(f"Failed to re-initialize BigQuery client in context manager: {e}")
                raise ConnectionError(f"BigQuery client re-initialization failed: {e}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_connection()


    def ping(self) -> bool:
        """Checks if the BigQuery service is accessible by trying to list datasets (limited)."""
        if not self.client:
            return False
        try:
            # Listing datasets is a simple API call to check connectivity and auth.
            datasets = list(self.client.list_datasets(max_results=1))
            logger.info("BigQuery connection ping successful (listed datasets).")
            return True
        except GoogleCloudError as e:
            logger.error(f"BigQuery connection ping failed: {e}")
            return False
        except Exception as e: # Catch other potential issues
            logger.error(f"BigQuery ping failed unexpectedly: {e}")
            return False

    def execute_query(self, query: str, job_config: Optional[bigquery.QueryJobConfig] = None, fetch_all: bool = True) -> Optional[List[Dict[str, Any]]]:
        """
        Executes a SQL query in BigQuery.

        Args:
            query (str): The SQL query to execute.
            job_config (Optional[bigquery.QueryJobConfig]): BigQuery job configuration.
                                                            Use this to set default_dataset, query_parameters, etc.
            fetch_all (bool): If True (default), fetches all rows. If False, returns the job object or row count.
                              Note: For DML, BigQuery returns information in the job object, not typically a row count from cursor.

        Returns:
            Optional[List[Dict[str, Any]]]: Query results as a list of dictionaries if fetch_all is True and query is a SELECT.
                                            Otherwise, returns the QueryJob object or None on error.
        """
        if not self.client:
            logger.error("BigQuery client not available for execute_query.")
            raise ConnectionError("BigQuery not connected.")

        final_job_config = job_config
        if not final_job_config and self.dataset_id:
            # If a default dataset is configured for the connector, use it.
            final_job_config = bigquery.QueryJobConfig(default_dataset=f"{self.client.project}.{self.dataset_id}")

        try:
            query_job = self.client.query(query, job_config=final_job_config)
            logger.info(f"Executed BigQuery query (Job ID: {query_job.job_id}). Waiting for results...")

            if fetch_all:
                # Wait for the job to complete and get results
                results_iterator = query_job.result() # This blocks until job is done
                # Convert rows to list of dicts
                results = [dict(row) for row in results_iterator]
                logger.info(f"Query completed. Fetched {len(results)} rows.")
                return results
            else:
                # For DML or if results are not immediately needed, return the job object.
                # Caller can then check job.state, job.num_dml_affected_rows, etc.
                logger.info(f"Query job submitted. State: {query_job.state}")
                return query_job # Or query_job.num_dml_affected_rows if that's more useful for DML

        except GoogleCloudError as e:
            logger.error(f"BigQuery query execution failed: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error executing BigQuery query: {e}", exc_info=True)
            raise

    # --- Placeholder methods for common operations ---
    def load_data_from_gcs(self, collection_name: str, gcs_uri: str, source_format: str = "CSV") -> Optional[str]:
        """Placeholder for loading data from GCS into a BigQuery table."""
        if not self.client or not self.dataset_id:
            logger.error("BigQuery client or dataset_id not configured for load_data_from_gcs.")
            return None

        table_id = f"{self.client.project}.{self.dataset_id}.{collection_name}"
        logger.info(f"Placeholder: load_data_from_gcs URI {gcs_uri} (format: {source_format}) into BigQuery table {table_id}")

        # Example:
        # job_config = bigquery.LoadJobConfig(source_format=bigquery.SourceFormat.CSV if source_format == "CSV" else source_format)
        # load_job = self.client.load_table_from_uri(gcs_uri, table_id, job_config=job_config)
        # load_job.result() # Wait for completion
        # return load_job.job_id
        return "mock_job_id_load"

    def get_table_schema(self, table_name: str, dataset_id: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """Placeholder for fetching table schema."""
        if not self.client: return None
        ds_id = dataset_id or self.dataset_id
        if not ds_id:
            logger.error("Dataset ID not provided or configured for get_table_schema.")
            return None

        table_id = f"{self.client.project}.{ds_id}.{table_name}"
        logger.info(f"Placeholder: get_table_schema for BigQuery table {table_id}")
        # Example:
        # try:
        #     table = self.client.get_table(table_id)
        #     return [{"name": field.name, "type": field.field_type, "mode": field.mode} for field in table.schema]
        # except NotFound:
        #     return None
        return [{"name": "column_a", "type": "STRING", "mode": "NULLABLE"}]

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # This example requires Google Cloud authentication to be set up in the environment
    # (e.g., via `gcloud auth application-default login` or GOOGLE_APPLICATION_CREDENTIALS env var)
    # and a valid project ID.

    class MockBigQueryConfig:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT") # Use environment's default project
        dataset_id = "your_test_dataset" # Replace with a dataset you can access or create

    test_bq_config = MockBigQueryConfig()

    if not test_bq_config.project_id:
        print("BigQuery example skipped: GOOGLE_CLOUD_PROJECT environment variable not set.")
    else:
        bq_connector = None
        try:
            print(f"Attempting to connect to BigQuery project: {test_bq_config.project_id}")
            # Using context manager
            with BigQueryConnector(config=test_bq_config) as bq_conn:
                if bq_conn.ping():
                    print(f"Successfully connected to BigQuery project '{bq_conn.client.project}'.")

                    # Ensure dataset exists (or handle creation)
                    try:
                        bq_conn.client.get_dataset(test_bq_config.dataset_id)
                        print(f"Dataset '{test_bq_config.dataset_id}' exists.")
                    except NotFound:
                        print(f"Dataset '{test_bq_config.dataset_id}' does not exist. Please create it or use an existing one for full testing.")
                        # Example: bq_conn.client.create_dataset(test_bq_config.dataset_id, exists_ok=True)
                        # print(f"Attempted to create dataset '{test_bq_config.dataset_id}'.")


                    # Example: Simple query
                    query = f"SELECT CURRENT_TIMESTAMP() as now;"
                    print(f"\nExecuting query: {query}")
                    results = bq_conn.execute_query(query, fetch_all=True)
                    if results:
                        print("Query Result:")
                        for row in results:
                            print(row)

                    # Example: Query with specific dataset (if not default in QueryJobConfig)
                    # public_dataset_query = "SELECT name FROM `bigquery-public-data.usa_names.usa_1910_current` WHERE name = 'Emma' LIMIT 5"
                    # print(f"\nExecuting query on public dataset: {public_dataset_query}")
                    # public_results = bq_conn.execute_query(public_dataset_query, fetch_all=True)
                    # if public_results:
                    #     print("Public Dataset Query Result:")
                    #     for row in public_results:
                    #         print(row)

                else:
                    print(f"Failed to ping BigQuery with project '{test_bq_config.project_id}'.")

        except ConnectionError as ce:
             print(f"BigQuery Connection Error: {ce}")
        except GoogleCloudError as ge:
            print(f"Google Cloud Error with BigQuery: {ge}")
        except Exception as e:
            print(f"An error occurred with BigQuery: {e}", exc_info=True)
