import pymongo
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
import logging
from typing import Optional, List, Dict, Any, Union

# Assuming MongoConfig is defined in config.manager, but to avoid direct import issues here for now:
# from ..config.manager import MongoConfig # Use this if structure allows
# For now, we'll assume config is passed as a dictionary or an object with necessary attributes.

logger = logging.getLogger(__name__)

class MongoDBConnector:
    def __init__(self, config: Any): # config should be compatible with MongoConfig
        self.connection_uri = getattr(config, 'connection_uri', "mongodb://localhost:27017/")
        self.database_name = getattr(config, 'database', 'rag_db_mongo')
        self.server_selection_timeout_ms = getattr(config, 'server_selection_timeout_ms', 5000)

        self.client: Optional[MongoClient] = None
        self.db: Optional[pymongo.database.Database] = None # Using pymongo.database.Database for type hint

        try:
            self.client = MongoClient(
                self.connection_uri,
                serverSelectionTimeoutMS=self.server_selection_timeout_ms
            )
            # The ismaster command is cheap and does not require auth.
            self.client.admin.command('ismaster') # Or self.client.admin.command('ping')
            self.db = self.client[self.database_name]
            logger.info(f"Successfully connected to MongoDB at {self.connection_uri} and selected database '{self.database_name}'.")
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {e}", exc_info=True)
            self.client = None
            self.db = None
            raise # Re-raise to indicate connection failure
        except Exception as e: # Catch other potential errors during init e.g. config issues
            logger.error(f"Failed to initialize MongoDB connector: {e}", exc_info=True)
            self.client = None
            self.db = None
            raise

    def close_connection(self):
        """Closes the MongoDB connection."""
        if self.client:
            try:
                self.client.close()
                logger.info("MongoDB connection closed.")
            except Exception as e:
                logger.error(f"Error closing MongoDB connection: {e}", exc_info=True)
            finally:
                self.client = None
                self.db = None

    def __enter__(self):
        # Connection is established in __init__. If it failed, self.db would be None.
        if not self.db: # Attempt to reconnect if not connected, or raise
            try:
                self.client = MongoClient(
                    self.connection_uri,
                    serverSelectionTimeoutMS=self.server_selection_timeout_ms
                )
                self.client.admin.command('ismaster')
                self.db = self.client[self.database_name]
                logger.info("Reconnected to MongoDB within context manager.")
            except Exception as e:
                logger.error(f"Failed to reconnect to MongoDB in context manager: {e}")
                raise ConnectionError(f"MongoDB not available: {e}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_connection()

    def ping(self) -> bool:
        """Checks if the MongoDB server is accessible."""
        if not self.client:
            return False
        try:
            self.client.admin.command('ping')
            logger.info("MongoDB ping successful.")
            return True
        except ConnectionFailure:
            logger.error("MongoDB ping failed: ConnectionFailure.")
            return False
        except Exception as e:
            logger.error(f"MongoDB ping failed: {e}", exc_info=True)
            return False

    # --- CRUD Operations ---
    def insert_document(self, collection_name: str, document: Dict[str, Any]) -> Optional[Any]:
        """Inserts a single document into the specified collection."""
        if not self.db:
            logger.error("MongoDB database not available for insert_document.")
            raise ConnectionError("MongoDB not connected.")
        try:
            collection = self.db[collection_name]
            result = collection.insert_one(document)
            logger.info(f"Inserted document into '{collection_name}' with ID: {result.inserted_id}")
            return result.inserted_id
        except OperationFailure as e:
            logger.error(f"MongoDB insert_document failed for collection '{collection_name}': {e}", exc_info=True)
            raise # Re-raise to allow specific handling
        except Exception as e:
            logger.error(f"Unexpected error in insert_document for collection '{collection_name}': {e}", exc_info=True)
            raise

    def find_documents(self, collection_name: str, query: Dict[str, Any], projection: Optional[Dict[str, Any]] = None, limit: int = 0) -> List[Dict[str, Any]]:
        """Finds documents in a collection matching the query."""
        if not self.db:
            logger.error("MongoDB database not available for find_documents.")
            raise ConnectionError("MongoDB not connected.")
        try:
            collection = self.db[collection_name]
            if limit > 0:
                cursor = collection.find(query, projection).limit(limit)
            else:
                cursor = collection.find(query, projection)
            results = list(cursor) # Evaluate cursor
            logger.debug(f"Found {len(results)} documents in '{collection_name}' matching query.")
            return results
        except OperationFailure as e:
            logger.error(f"MongoDB find_documents failed for collection '{collection_name}': {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in find_documents for collection '{collection_name}': {e}", exc_info=True)
            raise

    def find_document_by_id(self, collection_name: str, document_id: Any, projection: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Finds a single document by its _id."""
        if not self.db:
            logger.error("MongoDB database not available for find_document_by_id.")
            raise ConnectionError("MongoDB not connected.")
        try:
            from bson.objectid import ObjectId # Standard way to handle MongoDB _id
            if not isinstance(document_id, ObjectId):
                try: # Attempt to convert if it's a string that looks like an ObjectId
                    document_id = ObjectId(str(document_id))
                except Exception:
                    logger.warning(f"Document ID '{document_id}' is not a valid ObjectId string. Querying as is.")

            collection = self.db[collection_name]
            result = collection.find_one({"_id": document_id}, projection)
            logger.debug(f"find_document_by_id result for ID '{document_id}' in '{collection_name}': {'Found' if result else 'Not Found'}")
            return result
        except OperationFailure as e:
            logger.error(f"MongoDB find_document_by_id failed for collection '{collection_name}': {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in find_document_by_id for collection '{collection_name}': {e}", exc_info=True)
            raise

    def update_document(self, collection_name: str, query: Dict[str, Any], update_data: Dict[str, Any], upsert: bool = False) -> int:
        """Updates one or more documents matching the query."""
        if not self.db:
            logger.error("MongoDB database not available for update_document.")
            raise ConnectionError("MongoDB not connected.")
        try:
            collection = self.db[collection_name]
            # Use $set to avoid replacing the whole document unless intended
            if not any(key.startswith('$') for key in update_data.keys()):
                update_payload = {"$set": update_data}
            else:
                update_payload = update_data

            result = collection.update_many(query, update_payload, upsert=upsert)
            logger.info(f"Updated {result.modified_count} documents (matched {result.matched_count}, upserted_id {result.upserted_id}) in '{collection_name}'.")
            return result.modified_count
        except OperationFailure as e:
            logger.error(f"MongoDB update_document failed for collection '{collection_name}': {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in update_document for collection '{collection_name}': {e}", exc_info=True)
            raise

    def delete_documents(self, collection_name: str, query: Dict[str, Any]) -> int:
        """Deletes documents matching the query."""
        if not self.db:
            logger.error("MongoDB database not available for delete_documents.")
            raise ConnectionError("MongoDB not connected.")
        try:
            collection = self.db[collection_name]
            result = collection.delete_many(query)
            logger.info(f"Deleted {result.deleted_count} documents from '{collection_name}'.")
            return result.deleted_count
        except OperationFailure as e:
            logger.error(f"MongoDB delete_documents failed for collection '{collection_name}': {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in delete_documents for collection '{collection_name}': {e}", exc_info=True)
            raise

# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO) # For standalone test

    class MockMongoConfig:
        connection_uri = "mongodb://localhost:27017/" # Replace if your MongoDB is elsewhere or needs auth
        database = "rag_system_test_db"
        server_selection_timeout_ms = 2000 # Shorter timeout for local test

    test_mongo_config = MockMongoConfig()
    mongo_connector = None

    try:
        print("Attempting to connect to MongoDB...")
        with MongoDBConnector(config=test_mongo_config) as connector:
            mongo_connector = connector # Keep reference for potential use after 'with' if needed, though 'with' handles close
            if connector.ping():
                print(f"Successfully connected to MongoDB database '{connector.database_name}'.")

                test_collection = "test_items_mongo"

                # Insert
                item_id = connector.insert_document(test_collection, {"name": "MongoTestItem", "value": 100, "tags": ["test", "mongo"]})
                print(f"Inserted item with ID: {item_id}")

                if item_id:
                    # Find by ID
                    item = connector.find_document_by_id(test_collection, item_id)
                    print(f"Found item by ID: {item}")

                    # Update
                    updated_count = connector.update_document(test_collection, {"_id": item_id}, {"value": 150, "status": "updated"})
                    print(f"Updated {updated_count} item(s).")
                    item_updated = connector.find_document_by_id(test_collection, item_id)
                    print(f"Item after update: {item_updated}")

                # Find multiple
                items = connector.find_documents(test_collection, {"tags": "test"}, limit=5)
                print(f"Found items with 'test' tag: {items}")

                # Delete
                if item_id: # only delete if we successfully inserted
                    deleted_count = connector.delete_documents(test_collection, {"_id": item_id})
                    print(f"Deleted {deleted_count} item(s).")
                    item_deleted = connector.find_document_by_id(test_collection, item_id)
                    print(f"Item after delete: {item_deleted}")

    except ConnectionError as ce: # Catch ConnectionError from our wrapper
        print(f"MongoDB Connection Error (from wrapper): {ce}")
    except ConnectionFailure as cf: # Catch pymongo's ConnectionFailure
        print(f"MongoDB Connection Failure (pymongo): {cf}")
    except Exception as e:
        print(f"An error occurred with MongoDB: {e}", exc_info=True)

    # Note: if mongo_connector was used outside 'with', ensure close_connection is called
    # if mongo_connector and mongo_connector.client:
    #     mongo_connector.close_connection()
