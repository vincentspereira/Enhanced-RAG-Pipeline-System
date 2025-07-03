import os
import logging
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Driver, Result, Transaction, unit_of_work
from neo4j.exceptions import Neo4jError, ServiceUnavailable

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
        logger.error(f"Critical: Failed to import get_config_value for Neo4jConnector. Error: {e}")
        def get_config_value(env_var_name, yaml_path=None, default=None):
            return os.getenv(env_var_name, default)

class Neo4jConnector:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Neo4jConnector, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.uri = get_config_value("NEO4J_URI", yaml_path="graph_db.neo4j.uri", default="bolt://localhost:7687")
        self.user = get_config_value("NEO4J_USER", yaml_path="graph_db.neo4j.user", default="neo4j")
        self.password = get_config_value("NEO4J_PASSWORD", yaml_path="graph_db.neo4j.password") # Required

        self.driver: Optional[Driver] = None
        self._initialized = False

        if not self.password: # Password is required for Neo4j
            logger.error("Neo4j password not configured. Connector will not initialize.")
            return

        self._connect()

    def _connect(self):
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Verify connection by trying to get server info or a simple query
            with self.driver.session() as session:
                session.run("RETURN 1").consume() # Consume to ensure query executes
            logger.info(f"Successfully connected to Neo4j at {self.uri} as user {self.user}")
            self._initialized = True
        except ServiceUnavailable as su_err:
            logger.error(f"Neo4j service unavailable at {self.uri}: {su_err}", exc_info=True)
            self.driver = None
            self._initialized = False
        except Neo4jError as e:
            logger.error(f"Neo4j connection or authentication error: {e}", exc_info=True)
            self.driver = None
            self._initialized = False
        except Exception as e: # Catch any other potential errors during init
            logger.error(f"An unexpected error occurred connecting to Neo4j: {e}", exc_info=True)
            self.driver = None
            self._initialized = False

    def get_driver(self) -> Optional[Driver]:
        if not self._initialized or not self.driver:
            logger.warning("Neo4j driver not initialized or connection failed. Attempting to reconnect.")
            self._connect()
        if not self.driver:
             logger.error("Failed to establish Neo4j driver connection.")
        return self.driver

    def close(self):
        """Closes the Neo4j driver connection."""
        if self.driver:
            try:
                self.driver.close()
                logger.info("Neo4j connection closed.")
            except Neo4jError as e:
                logger.error(f"Error closing Neo4j connection: {e}", exc_info=True)
            finally:
                self.driver = None
                self._initialized = False

    def execute_read_query(self, cypher_query: str, params: Optional[Dict[str, Any]] = None) -> Optional[List[Dict[str, Any]]]:
        """Executes a read-only Cypher query."""
        driver = self.get_driver()
        if not driver:
            return None

        records_list = []
        try:
            with driver.session() as session:
                result: Result = session.read_transaction(self._run_cypher_query, cypher_query, params)
                for record in result:
                    records_list.append(dict(record)) # Convert Record object to dict
            return records_list
        except Neo4jError as e:
            logger.error(f"Error executing Neo4j read query: {cypher_query} with params {params}. Error: {e}", exc_info=True)
            return None

    def execute_write_query(self, cypher_query: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Executes a write Cypher query and returns summary or specific results."""
        driver = self.get_driver()
        if not driver:
            return None

        try:
            with driver.session() as session:
                result: Result = session.write_transaction(self._run_cypher_query, cypher_query, params)
                # For writes, you might want to return summary, or if query returns data, that data.
                # Here, we'll return the summary. If the query has a RETURN clause, result will contain records.
                # If you want to process returned records, you'd iterate through `result` like in read_query.
                return result.consume().summary # Consume to get summary
        except Neo4jError as e:
            logger.error(f"Error executing Neo4j write query: {cypher_query} with params {params}. Error: {e}", exc_info=True)
            return None

    @staticmethod
    @unit_of_work(timeout=30.0) # Example: set a timeout for the transaction
    def _run_cypher_query(tx: Transaction, cypher_query: str, params: Optional[Dict[str, Any]] = None) -> Result:
        """Helper function to run a Cypher query within a transaction."""
        logger.debug(f"Executing Cypher: {cypher_query[:200]}... with params: {params}")
        return tx.run(cypher_query, params or {})

    # --- Higher-level helper methods (examples) ---

    def create_node(self, label: str, properties: Dict[str, Any]) -> Optional[Any]:
        """Creates a node with a given label and properties."""
        # Cypher for MERGE to avoid duplicates if a unique property is known, or CREATE for new.
        # Example: Assuming 'name' is a unique property for this label.
        # query = f"MERGE (n:{label} {{name: $name}}) SET n += $props RETURN id(n) AS node_id"
        # params = {"name": properties.get("name"), "props": properties}
        # If no single unique property, use CREATE:
        query = f"CREATE (n:{label} $props) RETURN id(n) AS node_id"
        params = {"props": properties}

        summary = self.execute_write_query(query, params)
        if summary and summary.counters.nodes_created == 1:
            # If you need the ID, you'd parse it from a RETURN clause.
            # For simplicity, just confirming creation.
            logger.info(f"Node with label '{label}' and properties {properties} created.")
            return summary # Or a specific ID if returned and parsed
        elif summary: # MERGE might result in nodes_created = 0 if it matched
             logger.info(f"Node with label '{label}' and properties {properties} merged/found. Nodes created: {summary.counters.nodes_created}")
             return summary
        return None

    def create_relationship(self, start_node_label: str, start_node_props: Dict[str, Any],
                            end_node_label: str, end_node_props: Dict[str, Any],
                            relationship_type: str, rel_props: Optional[Dict[str, Any]] = None) -> bool:
        """
        Creates a relationship between two nodes.
        Nodes are matched based on one unique property (e.g., 'name' or 'id').
        """
        # Assuming 'name' is the property to match nodes. Adjust as needed.
        # This query is simplified; robust matching might require specific unique IDs.
        query = f"""
        MATCH (a:{start_node_label} {{name: $start_name}}), (b:{end_node_label} {{name: $end_name}})
        MERGE (a)-[r:{relationship_type}]->(b)
        """
        if rel_props:
            query += " SET r += $rel_props"

        params = {
            "start_name": start_node_props.get("name"),
            "end_name": end_node_props.get("name"),
            "rel_props": rel_props or {}
        }

        summary = self.execute_write_query(query, params)
        if summary and summary.counters.relationships_created >= 0: # Can be 0 if relationship already existed via MERGE
            logger.info(f"Relationship '{relationship_type}' created/merged between nodes. Relationships created: {summary.counters.relationships_created}")
            return True
        return False


# Example Usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # IMPORTANT: For this example to run, you MUST:
    # 1. Have a Neo4j instance running (e.g., via Docker: `docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/yourStrongPassword neo4j:latest`)
    # 2. Set NEO4J_PASSWORD environment variable. Optionally NEO4J_URI, NEO4J_USER.
    #    e.g., export NEO4J_PASSWORD="yourStrongPassword"

    if not os.getenv("NEO4J_PASSWORD"):
        logger.error("NEO4J_PASSWORD environment variable not set. Skipping example.")
    else:
        logger.info("Neo4j environment variables found. Running example...")
        neo_connector = Neo4jConnector()

        if neo_connector._initialized:
            # Test 1: Create some nodes
            logger.info("Test 1: Creating nodes...")
            summary_alice = neo_connector.create_node("Person", {"name": "Alice", "age": 30})
            summary_bob = neo_connector.create_node("Person", {"name": "Bob", "age": 32})
            summary_project = neo_connector.create_node("Project", {"name": "RAG System", "status": "active"})

            if summary_alice and summary_bob and summary_project:
                 logger.info("Nodes Alice, Bob, and Project RAG System created/merged.")
            else:
                 logger.error("Failed to create one or more initial nodes.")

            # Test 2: Create relationships
            logger.info("\nTest 2: Creating relationships...")
            if summary_alice and summary_project: # Ensure nodes were created
                rel1_success = neo_connector.create_relationship(
                    "Person", {"name": "Alice"},
                    "Project", {"name": "RAG System"},
                    "WORKS_ON", {"role": "Developer"}
                )
                if rel1_success: logger.info("Alice WORKS_ON RAG System relationship created/merged.")

            if summary_bob and summary_project:
                rel2_success = neo_connector.create_relationship(
                    "Person", {"name": "Bob"},
                    "Project", {"name": "RAG System"},
                    "MANAGES"
                )
                if rel2_success: logger.info("Bob MANAGES RAG System relationship created/merged.")

            if summary_alice and summary_bob:
                rel3_success = neo_connector.create_relationship(
                    "Person", {"name": "Alice"},
                    "Person", {"name": "Bob"},
                    "COLLEAGUE_OF"
                )
                if rel3_success: logger.info("Alice COLLEAGUE_OF Bob relationship created/merged.")


            # Test 3: Read data - Find people working on the RAG System
            logger.info("\nTest 3: Reading data...")
            query_people_on_project = """
            MATCH (p:Person)-[:WORKS_ON]->(proj:Project {name: $projectName})
            RETURN p.name AS personName, p.age AS personAge
            """
            results = neo_connector.execute_read_query(query_people_on_project, {"projectName": "RAG System"})
            if results is not None:
                logger.info(f"People working on 'RAG System': {results}")
            else:
                logger.error("Failed to execute read query for people on project.")

            # Test 4: Read data - Find all relationships of Alice
            query_alice_rels = """
            MATCH (p:Person {name: $personName})-[r]-(related)
            RETURN type(r) as relationshipType, related.name as relatedNodeName
            """
            alice_rels = neo_connector.execute_read_query(query_alice_rels, {"personName": "Alice"})
            if alice_rels is not None:
                logger.info(f"Alice's relationships: {alice_rels}")


            # Cleanup (optional): Delete test data
            logger.info("\nTest Cleanup: Deleting test data...")
            cleanup_query = "MATCH (n) WHERE n.name IN ['Alice', 'Bob', 'RAG System'] DETACH DELETE n"
            cleanup_summary = neo_connector.execute_write_query(cleanup_query)
            if cleanup_summary:
                logger.info(f"Cleanup successful. Nodes deleted: {cleanup_summary.counters.nodes_deleted}, Relationships deleted: {cleanup_summary.counters.relationships_deleted}")
            else:
                logger.error("Cleanup query failed.")

            neo_connector.close()
        else:
            logger.error("Neo4jConnector failed to initialize. Example not run.")
            logger.error("Ensure Neo4j is running and NEO4J_PASSWORD (and optionally URI/USER) are correctly set.")
