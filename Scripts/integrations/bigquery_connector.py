import os
from google.cloud import bigquery
from google.oauth2 import service_account # For service account key file
from google.api_core.exceptions import GoogleAPICallError
import logging
from typing import List, Dict, Any, Optional, Iterator

# Configure logger
logger = logging.getLogger(__name__)

# Import config loader
try:
    from Scripts.utils.config_loader import get_config_value
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from utils.config_loader import get_config_value
    except ImportError as e:
        logger.error(f"Critical: Failed to import get_config_value for BigQueryConnector. Error: {e}")
        def get_config_value(env_var_name, yaml_path=None, default=None): # Basic fallback
            return os.getenv(env_var_name, default)

class BigQueryConnector:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(BigQueryConnector, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.project_id = get_config_value("BIGQUERY_PROJECT_ID", yaml_path="data_warehouse.bigquery.project_id")
        self.service_account_json_path = get_config_value("GOOGLE_APPLICATION_CREDENTIALS", yaml_path="data_warehouse.bigquery.service_account_json_path") # Standard env var for gcloud
        self.location = get_config_value("BIGQUERY_LOCATION", yaml_path="data_warehouse.bigquery.location", default="US") # e.g., "US", "EU", "asia-northeast1"


        self.client = None
        self._initialized = False

        if not self.project_id:
            logger.error("BigQuery Project ID not configured. Connector will not initialize.")
            return

        self._connect()

    def _connect(self):
        try:
            if self.service_account_json_path:
                # Explicitly use service account credentials if path is provided
                if not os.path.exists(self.service_account_json_path):
                    logger.error(f"Service account key file not found at: {self.service_account_json_path}. BigQuery client will use Application Default Credentials if available.")
                    # Fallback to ADC by not passing credentials
                    self.client = bigquery.Client(project=self.project_id, location=self.location)
                else:
                    credentials = service_account.Credentials.from_service_account_file(self.service_account_json_path)
                    self.client = bigquery.Client(project=self.project_id, credentials=credentials, location=self.location)
                    logger.info(f"BigQuery client initialized using service account: {self.service_account_json_path}")
            else:
                # Use Application Default Credentials (ADC)
                # This works if gcloud SDK is configured, or running in GCP environment (e.g., GCE, GKE, Cloud Functions)
                self.client = bigquery.Client(project=self.project_id, location=self.location)
                logger.info("BigQuery client initialized using Application Default Credentials (ADC).")

            # Test connection by listing datasets (lightweight operation)
            # list(self.client.list_datasets(max_results=1)) # This actually makes an API call
            logger.info(f"Successfully initialized BigQuery client for project: {self.project_id}, location: {self.location}")
            self._initialized = True
        except GoogleAPICallError as e:
            logger.error(f"Google API Call Error during BigQuery client initialization: {e}", exc_info=True)
            self.client = None
            self._initialized = False
        except Exception as e:
            logger.error(f"An unexpected error occurred initializing BigQuery client: {e}", exc_info=True)
            self.client = None
            self._initialized = False

    def get_client(self) -> Optional[bigquery.Client]:
        if not self._initialized or not self.client:
            logger.warning("BigQuery client not initialized. Attempting to reconnect.")
            self._connect()

        if not self.client: # If reconnect failed
             logger.error("Failed to establish BigQuery client connection.")
        return self.client

    def execute_query(self, query: str, query_params: Optional[List[Any]] = None) -> Optional[Iterator[bigquery.table.Row]]:
        """
        Executes a SQL query on BigQuery.
        Args:
            query (str): The SQL query to execute.
            query_params (Optional[List[Any]]): A list of query parameters for positional params (?),
                                                or dict for named params (@name).
                                                For this stub, we'll focus on no params or simple positional.
        Returns:
            An iterator of bigquery.table.Row objects, or None on error.
        """
        client = self.get_client()
        if not client:
            return None

        try:
            job_config = bigquery.QueryJobConfig()
            if query_params:
                # Example for positional parameters
                # For named: job_config.query_parameters = [bigquery.ScalarQueryParameter("name", "STRING", value)]
                # This part needs to be more robust based on how params are passed.
                # For now, assume simple list of BQ ScalarQueryParameter objects if using params.
                # Or, if params are just for f-string style replacement (NOT RECOMMENDED for SQLi risk),
                # then it's handled before calling this.
                # For this stub, let's assume query_params are already formatted for BQ if used.
                # A better implementation would construct BQ QueryParameter objects.
                # For now, assuming no complex params or they are handled by the caller constructing the query string.
                pass # Placeholder for query parameter handling if needed.

            logger.info(f"Executing BigQuery query (first 100 chars): {query[:100]}...")
            query_job = client.query(query, job_config=job_config, location=self.location) # API request

            # Wait for the job to complete (optional, can iterate directly for large results)
            # query_job.result() # This blocks until job is done

            logger.info(f"BigQuery job {query_job.job_id} started. State: {query_job.state}")
            return query_job.result() # Returns an iterator of Row objects

        except GoogleAPICallError as e:
            logger.error(f"Google API Call Error executing BigQuery query: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error executing BigQuery query: {e}", exc_info=True)
            return None

    def close_client(self):
        """BigQuery client typically doesn't need explicit closing for resource management
           as it uses HTTP-based APIs. Connections are managed by the underlying HTTP library.
           This method is a placeholder if specific cleanup were ever needed.
        """
        if self.client:
            # self.client.close() # bigquery.Client doesn't have a close() method.
            logger.info("BigQuery client does not require explicit closing. Resources managed by Google Cloud library.")
            self.client = None # Allow re-initialization if needed
        self._initialized = False


# Example Usage (for testing purposes - requires GCP project with BigQuery API enabled and auth)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # --- IMPORTANT ---
    # For this example to run, you MUST:
    # 1. Set `BIGQUERY_PROJECT_ID` environment variable or in config.yaml.
    # 2. Have Application Default Credentials configured (e.g., by running `gcloud auth application-default login`)
    #    OR set `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of a service account JSON key file.
    #    OR update `config.yaml` with `data_warehouse.bigquery.service_account_json_path`.

    logger.info("Attempting to connect to BigQuery (ensure environment/config is set for auth and project ID)...")

    # Example: Create a dummy config.yaml for this test
    # with open("config.yaml", "w") as f:
    #     f.write("""
    # data_warehouse:
    #   bigquery:
    #     project_id: "your-gcp-project-id"
    #     # service_account_json_path: "/path/to/your/service-account-key.json" # Optional
    #     # location: "US" # Optional
    # """)

    bq_connector = BigQueryConnector()

    if bq_connector._initialized and bq_connector.client:
        logger.info("BigQueryConnector initialized successfully.")

        # Test 1: Simple query using a public dataset
        # This query might take a moment and incur small BigQuery costs if run.
        query1 = """
            SELECT corpus, COUNT(word) AS word_count
            FROM `bigquery-public-data.samples.shakespeare`
            WHERE word = 'king'
            GROUP BY corpus
            ORDER BY word_count DESC
            LIMIT 5;
        """
        logger.info(f"Executing test query on public dataset: {query1}")
        results_iterator = bq_connector.execute_query(query1)

        if results_iterator:
            logger.info("Query results for 'king' in Shakespeare samples:")
            results_list = list(results_iterator) # Convert iterator to list to see results
            if results_list:
                for row in results_list:
                    logger.info(f"  Corpus: {row.corpus}, Word Count: {row.word_count}")
            else:
                logger.info("  Query returned no results (this might be unexpected for the sample query).")
        else:
            logger.info("Test query 1 failed or returned no iterator.")

        # Test 2: Query a non-existent table to see error handling (optional)
        # query_error = "SELECT * FROM `your-project.your_dataset.non_existent_table` LIMIT 10;"
        # logger.info(f"Executing query expected to fail: {query_error}")
        # error_results = bq_connector.execute_query(query_error)
        # if error_results is None:
        #     logger.info("Query correctly failed as expected (returned None).")
        # else:
        #     logger.warning(f"Query for non-existent table unexpectedly returned an iterator: {list(error_results)}")


        bq_connector.close_client() # Placeholder, doesn't do much for BQ client
    else:
        logger.error("BigQueryConnector failed to initialize. Check configurations, credentials, and GCP project settings.")
        logger.error("Ensure BIGQUERY_PROJECT_ID is set and ADC or GOOGLE_APPLICATION_CREDENTIALS are correctly configured.")

    # Clean up dummy config if created
    # if os.path.exists("config.yaml"):
    #     os.remove("config.yaml")
