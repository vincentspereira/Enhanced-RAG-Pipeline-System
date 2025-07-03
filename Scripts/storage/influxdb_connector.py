import os
import logging
from typing import List, Dict, Any, Optional
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS
from influxdb_client.client.exceptions import InfluxDBError

logger = logging.getLogger(__name__)

# Try to import get_config_value, fallback if necessary
try:
    from Scripts.utils.config_loader import get_config_value
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        from utils.config_loader import get_config_value
    except ImportError as e:
        logger.error(f"Critical: Failed to import get_config_value for InfluxDBConnector. Error: {e}")
        def get_config_value(env_var_name, yaml_path=None, default=None):
            return os.getenv(env_var_name, default)

class InfluxDBConnector:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(InfluxDBConnector, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.url = get_config_value("INFLUXDB_URL", yaml_path="time_series_db.influxdb.url", default="http://localhost:8086")
        self.token = get_config_value("INFLUXDB_TOKEN", yaml_path="time_series_db.influxdb.token") # Required
        self.org = get_config_value("INFLUXDB_ORG", yaml_path="time_series_db.influxdb.org")       # Required
        self.default_bucket = get_config_value("INFLUXDB_BUCKET", yaml_path="time_series_db.influxdb.default_bucket", default="my-bucket")

        self.client: Optional[InfluxDBClient] = None
        self.write_api = None
        self.query_api = None
        self._initialized = False

        if not all([self.token, self.org]):
            logger.error("InfluxDB token or organization not configured. Connector will not initialize.")
            return

        self._connect()

    def _connect(self):
        try:
            self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
            # Check connectivity
            if self.client.ping():
                self.write_api = self.client.write_api(write_options=SYNCHRONOUS) # Or ASYNCHRONOUS for performance
                self.query_api = self.client.query_api()
                logger.info(f"Successfully connected to InfluxDB at {self.url}, org: {self.org}")
                self._initialized = True
            else:
                logger.error(f"Failed to ping InfluxDB at {self.url}. Check URL, token, and org.")
                self.client = None # Ensure client is None if ping fails
                self._initialized = False
        except InfluxDBError as e:
            logger.error(f"InfluxDB connection or configuration error: {e}", exc_info=True)
            self.client = None
            self._initialized = False
        except Exception as e:
            logger.error(f"An unexpected error occurred connecting to InfluxDB: {e}", exc_info=True)
            self.client = None
            self._initialized = False

    def get_client(self) -> Optional[InfluxDBClient]:
        if not self._initialized or not self.client:
            logger.warning("InfluxDB client not initialized or connection failed. Attempting to reconnect.")
            self._connect()
        if not self.client:
             logger.error("Failed to establish InfluxDB client connection.")
        return self.client

    def write_point(self, measurement: str, tags: Dict[str, str], fields: Dict[str, Any],
                    timestamp: Optional[Any] = None, bucket: Optional[str] = None,
                    precision: WritePrecision = WritePrecision.NS):
        """Writes a single data point to InfluxDB."""
        if not self.write_api:
            logger.error("Write API not initialized. Cannot write point.")
            if not self._initialized: self._connect() # Try to reconnect if not initialized
            if not self.write_api: return False # If still not initialized after reconnect attempt

        _bucket = bucket or self.default_bucket
        point = Point(measurement)
        for tag_key, tag_value in tags.items():
            point.tag(tag_key, tag_value)
        for field_key, field_value in fields.items():
            point.field(field_key, field_value)
        if timestamp:
            point.time(timestamp, precision)

        try:
            self.write_api.write(bucket=_bucket, org=self.org, record=point)
            logger.debug(f"Successfully wrote point to measurement '{measurement}' in bucket '{_bucket}'.")
            return True
        except InfluxDBError as e:
            logger.error(f"Error writing point to InfluxDB: {e}", exc_info=True)
            return False

    def write_points(self, records: List[Point], bucket: Optional[str] = None):
        """Writes multiple data points to InfluxDB."""
        if not self.write_api:
            logger.error("Write API not initialized. Cannot write points.")
            if not self._initialized: self._connect()
            if not self.write_api: return False

        _bucket = bucket or self.default_bucket
        try:
            self.write_api.write(bucket=_bucket, org=self.org, record=records)
            logger.debug(f"Successfully wrote {len(records)} points to bucket '{_bucket}'.")
            return True
        except InfluxDBError as e:
            logger.error(f"Error writing points to InfluxDB: {e}", exc_info=True)
            return False

    def query_data(self, flux_query: str) -> Optional[List[Any]]:
        """Executes a Flux query and returns the results."""
        if not self.query_api:
            logger.error("Query API not initialized. Cannot execute query.")
            if not self._initialized: self._connect()
            if not self.query_api: return None

        try:
            logger.debug(f"Executing Flux query: {flux_query[:200]}...")
            tables = self.query_api.query(query=flux_query, org=self.org)
            results = []
            for table in tables:
                for record in table.records:
                    results.append(record.values) # record.values is a dictionary
            return results
        except InfluxDBError as e:
            logger.error(f"Error executing Flux query on InfluxDB: {e}", exc_info=True)
            return None
        except Exception as e: # Catch other potential errors, e.g. network issues during query
            logger.error(f"Unexpected error during Flux query: {e}", exc_info=True)
            return None

    def close_connection(self):
        """Closes the InfluxDB client."""
        if self.client:
            try:
                self.client.close()
                logger.info("InfluxDB connection closed.")
            except InfluxDBError as e:
                logger.error(f"Error closing InfluxDB connection: {e}", exc_info=True)
            finally:
                self.client = None
                self.write_api = None
                self.query_api = None
                self._initialized = False

# Example Usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # IMPORTANT: For this example to run, you MUST:
    # 1. Have an InfluxDB v2.x instance running and accessible.
    # 2. Set the following environment variables:
    #    INFLUXDB_URL (e.g., "http://localhost:8086")
    #    INFLUXDB_TOKEN (your InfluxDB API token)
    #    INFLUXDB_ORG (your InfluxDB organization name)
    #    INFLUXDB_BUCKET (a bucket you have write/read access to, e.g., "test_bucket")
    #    e.g. export INFLUXDB_TOKEN="mySuPeRSeCreTtOkEn"

    if not all(os.getenv(var) for var in ["INFLUXDB_URL", "INFLUXDB_TOKEN", "INFLUXDB_ORG", "INFLUXDB_BUCKET"]):
        logger.error("One or more InfluxDB environment variables (INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG, INFLUXDB_BUCKET) are not set. Skipping example.")
    else:
        logger.info("InfluxDB environment variables found. Running example...")
        influx_connector = InfluxDBConnector()

        if influx_connector._initialized:
            # Test 1: Write a single point
            logger.info("Test 1: Writing a single point...")
            tags1 = {"host": "server01", "region": "us-west"}
            fields1 = {"cpu_load": 0.65, "memory_usage": 2048}
            success1 = influx_connector.write_point("system_metrics", tags1, fields1, bucket=influx_connector.default_bucket)
            if success1:
                logger.info("Single point written successfully.")
            else:
                logger.error("Failed to write single point.")

            # Test 2: Write multiple points
            logger.info("\nTest 2: Writing multiple points...")
            point1 = Point("iot_sensor").tag("device_id", "temp001").field("temperature", 23.5).time(time.time_ns(), WritePrecision.NS)
            point2 = Point("iot_sensor").tag("device_id", "humid002").field("humidity", 45.2).time(time.time_ns() - 10**9, WritePrecision.NS) # 1 second ago
            success2 = influx_connector.write_points([point1, point2], bucket=influx_connector.default_bucket)
            if success2:
                logger.info("Multiple points written successfully.")
            else:
                logger.error("Failed to write multiple points.")

            # Allow some time for data to be processed by InfluxDB if write_api is ASYNCHRONOUS
            # (though current default is SYNCHRONOUS)
            time.sleep(1)

            # Test 3: Query data
            logger.info("\nTest 3: Querying data...")
            # Query for the single point written in Test 1 (adjust time range if needed)
            flux_q = f'''
                from(bucket: "{influx_connector.default_bucket}")
                  |> range(start: -5m)
                  |> filter(fn: (r) => r["_measurement"] == "system_metrics")
                  |> filter(fn: (r) => r["host"] == "server01")
                  |> yield(name: "last_cpu_load")
            '''
            query_results = influx_connector.query_data(flux_q)
            if query_results is not None:
                logger.info(f"Query results for 'system_metrics' (last 5m): {query_results}")
                if not query_results:
                    logger.warning("Query returned no results. Data might not be written or time range is too narrow.")
            else:
                logger.error("Flux query failed or returned None.")

            # Query for IoT sensor data
            flux_q_iot = f'''
                from(bucket: "{influx_connector.default_bucket}")
                  |> range(start: -1h)
                  |> filter(fn: (r) => r["_measurement"] == "iot_sensor")
                  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                  |> limit(n:10)
            '''
            iot_results = influx_connector.query_data(flux_q_iot)
            if iot_results is not None:
                logger.info(f"Query results for 'iot_sensor' (last 1h, pivoted, limit 10): {iot_results}")
            else:
                logger.error("IoT Flux query failed or returned None.")

            influx_connector.close_connection()
        else:
            logger.error("InfluxDBConnector failed to initialize. Example not run.")
            logger.error("Ensure InfluxDB is running and INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG, INFLUXDB_BUCKET are correctly set.")
