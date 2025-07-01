import os
import pymongo
from pymongo.errors import ConnectionFailure, ConfigurationError
import logging

# Configure logger
logger = logging.getLogger(__name__)

# Import config loader
try:
    from Scripts.utils.config_loader import get_config_value
except ImportError:
    # Fallback for direct execution / PYTHONPATH issues
    import sys
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        from utils.config_loader import get_config_value
    except ImportError as e:
        logger.error(f"Critical: Failed to import get_config_value for MongoDB. Error: {e}")
        def get_config_value(env_var_name, yaml_path=None, default=None):
            return os.getenv(env_var_name, default)


class MongoDBConnector:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MongoDBConnector, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.client = None
        self.db = None

        # Configuration priority:
        # 1. MONGO_CONNECTION_STRING environment variable
        # 2. Individual MONGO_HOST, MONGO_PORT, etc. environment variables
        # 3. Values from config.yaml via get_config_value
        # 4. Hardcoded defaults

        connection_string = get_config_value("MONGO_CONNECTION_STRING", yaml_path="database.mongodb.connection_string")

        if connection_string:
            logger.info("Connecting to MongoDB using connection string.")
            self.connection_url = connection_string # Store for potential re-init
        else:
            logger.info("Constructing MongoDB connection URL from individual parameters.")
            self.host = get_config_value("MONGO_HOST", yaml_path="database.mongodb.host", default="localhost")
            self.port = int(get_config_value("MONGO_PORT", yaml_path="database.mongodb.port", default=27017))
            self.user = get_config_value("MONGO_USER", yaml_path="database.mongodb.user")
            self.password = get_config_value("MONGO_PASSWORD", yaml_path="database.mongodb.password")
            self.db_name = get_config_value("MONGO_DB", yaml_path="database.mongodb.dbname", default="mydatabase")

            if self.user and self.password:
                self.connection_url = f"mongodb://{self.user}:{self.password}@{self.host}:{self.port}/{self.db_name}?authSource=admin"
            else:
                self.connection_url = f"mongodb://{self.host}:{self.port}/{self.db_name}"

        try:
            self.client = pymongo.MongoClient(self.connection_url)
            # The ismaster command is cheap and does not require auth.
            self.client.admin.command('ismaster')
            logger.info(f"Successfully connected to MongoDB at {self.connection_url.split('@')[-1] if '@' in self.connection_url else self.connection_url}")

            # Determine DB name from connection string if not set by individual params
            if not hasattr(self, 'db_name') or not self.db_name:
                # Try to get db_name from connection string if possible (can be complex)
                # For simplicity, if connection_string is used, db_name should be part of it or a default is used.
                parsed_db_name = pymongo.uri_parser.parse_uri(self.connection_url).get('database')
                self.db_name = parsed_db_name if parsed_db_name else "default_db" # Fallback db name
                logger.info(f"Using database name: {self.db_name}")

            self.db = self.client[self.db_name]
            self._initialized = True
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {e}")
            self.client = None
            self.db = None
            self._initialized = False
        except ConfigurationError as e:
            logger.error(f"MongoDB configuration error: {e}")
            self.client = None
            self.db = None
            self._initialized = False
        except Exception as e: # Catch any other potential errors during init
            logger.error(f"An unexpected error occurred during MongoDB initialization: {e}")
            self.client = None
            self.db = None
            self._initialized = False


    def get_db(self):
        if not self.client or not self.db or not self._initialized:
            logger.warning("MongoDB client not initialized or connection failed. Attempting to reconnect...")
            # Attempt to re-initialize. Be careful with recursion if __init__ calls get_db.
            self.__init__()
            if not self.client or not self.db or not self._initialized:
                 logger.error("Failed to re-establish MongoDB connection.")
                 return None
        return self.db

    def get_collection(self, collection_name):
        db = self.get_db()
        if db:
            return db[collection_name]
        return None

    def close_connection(self):
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed.")
            self._initialized = False
            self.client = None
            self.db = None

    # --- Example CRUD Operations ---
    def insert_one(self, collection_name, document):
        collection = self.get_collection(collection_name)
        if collection is not None:
            try:
                result = collection.insert_one(document)
                logger.info(f"Inserted document with ID: {result.inserted_id} into {collection_name}")
                return result.inserted_id
            except Exception as e:
                logger.error(f"Error inserting document into {collection_name}: {e}")
        return None

    def find_one(self, collection_name, query):
        collection = self.get_collection(collection_name)
        if collection is not None:
            try:
                return collection.find_one(query)
            except Exception as e:
                logger.error(f"Error finding document in {collection_name}: {e}")
        return None

    def find_many(self, collection_name, query, projection=None):
        collection = self.get_collection(collection_name)
        if collection is not None:
            try:
                if projection:
                    return list(collection.find(query, projection))
                return list(collection.find(query))
            except Exception as e:
                logger.error(f"Error finding documents in {collection_name}: {e}")
        return [] # Return empty list on error or if collection is None

    def update_one(self, collection_name, query, update_document, upsert=False):
        collection = self.get_collection(collection_name)
        if collection is not None:
            try:
                result = collection.update_one(query, {"$set": update_document}, upsert=upsert)
                logger.info(f"Updated {result.modified_count} document(s) in {collection_name}. Upserted ID: {result.upserted_id}")
                return result
            except Exception as e:
                logger.error(f"Error updating document in {collection_name}: {e}")
        return None

    def delete_one(self, collection_name, query):
        collection = self.get_collection(collection_name)
        if collection is not None:
            try:
                result = collection.delete_one(query)
                logger.info(f"Deleted {result.deleted_count} document(s) from {collection_name}")
                return result.deleted_count
            except Exception as e:
                logger.error(f"Error deleting document from {collection_name}: {e}")
        return 0


# Example Usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Set environment variables for testing if not using config.yaml or actual env vars
    # os.environ["MONGO_HOST"] = "localhost"
    # os.environ["MONGO_DB"] = "testdb"
    # os.environ["MONGO_USER"] = "youruser" # if auth enabled
    # os.environ["MONGO_PASSWORD"] = "yourpassword" # if auth enabled
    # OR
    # os.environ["MONGO_CONNECTION_STRING"] = "mongodb://localhost:27017/testdb"

    mongo_connector = MongoDBConnector()

    if not mongo_connector._initialized:
        logger.error("MongoDBConnector not initialized. Exiting test.")
        exit()

    test_collection_name = "test_items_mongo"

    # Test 1: Insert a document
    doc_id = mongo_connector.insert_one(test_collection_name, {"name": "Mongo Test Item", "quantity": 50})
    if doc_id:
        logger.info(f"Test 1: Inserted item with ID: {doc_id}")

        # Test 2: Find the document
        item = mongo_connector.find_one(test_collection_name, {"_id": doc_id})
        if item:
            logger.info(f"Test 2: Found item: {item}")

        # Test 3: Update the document
        update_result = mongo_connector.update_one(test_collection_name, {"_id": doc_id}, {"quantity": 55, "status": "updated"})
        if update_result and update_result.modified_count > 0:
            logger.info("Test 3: Item updated successfully.")
            updated_item = mongo_connector.find_one(test_collection_name, {"_id": doc_id})
            logger.info(f"Test 3: Updated item data: {updated_item}")

        # Test 4: Find many (all in this case)
        all_items = mongo_connector.find_many(test_collection_name, {})
        logger.info(f"Test 4: All items in '{test_collection_name}': {all_items}")

        # Test 5: Delete the document
        deleted_count = mongo_connector.delete_one(test_collection_name, {"_id": doc_id})
        if deleted_count > 0:
            logger.info(f"Test 5: Item deleted successfully.")
            item_after_delete = mongo_connector.find_one(test_collection_name, {"_id": doc_id})
            logger.info(f"Test 5: Item after delete (should be None): {item_after_delete}")

        # Clean up the test collection (optional)
        # mongo_connector.get_collection(test_collection_name).drop()
        # logger.info(f"Dropped test collection: {test_collection_name}")

    mongo_connector.close_connection()
