"""
Knowledge graph integration module.
"""
from typing import List, Dict, Any, Optional
import os
import uuid
import hashlib

from Scripts.knowledge_graph.entity_extraction import EntityExtractor
from Scripts.knowledge_graph.graph_store import KnowledgeGraph

class KnowledgeGraphBuilder:
    """Builder for constructing and managing knowledge graphs."""
    
    def __init__(self, graph_name: str = "knowledge_graph",
                graph_path: Optional[str] = None,
                model_name: str = "en_core_web_lg"):
        """Initialize knowledge graph builder.
        
        Args:
            graph_name: Name of the knowledge graph
            graph_path: Path to save/load the graph
            model_name: spaCy model name for entity extraction
        """
        self.graph_name = graph_name
        self.graph_path = graph_path or f"{graph_name}.json"
        
        # Initialize graph
        if os.path.exists(self.graph_path):
            self.graph = KnowledgeGraph.load(self.graph_path)
        else:
            self.graph = KnowledgeGraph(name=graph_name)
            
        # Initialize entity extractor
        self.extractor = EntityExtractor(model_name=model_name)
        
        # Cache to avoid duplicate entities
        self.entity_cache = set()
        
    def process_document(self, document_id: str, document_text: str,
                        document_metadata: Optional[Dict[str, Any]] = None):
        """Process a document to extract entities and relationships.
        
        Args:
            document_id: ID of the document
            document_text: Text content of the document
            document_metadata: Additional metadata for the document
        """
        # Add document as an entity
        if document_metadata is None:
            document_metadata = {}
            
        self.graph.add_entity(
            entity_id=document_id,
            entity_type="Document",
            properties={
                "text": document_text[:1000] + "..." if len(document_text) > 1000 else document_text,
                **document_metadata
            }
        )
        
        # Extract entities
        entities = self.extractor.extract_entities(document_text)
        
        # Process each entity
        for entity in entities:
            # Generate entity ID
            entity_id = self._generate_entity_id(entity["text"], entity["type"])
            
            # Skip if already processed
            if entity_id in self.entity_cache:
                continue
                
            # Add entity to graph
            self.graph.add_entity(
                entity_id=entity_id,
                entity_type=entity["type"],
                properties={
                    "name": entity["text"],
                    "label": entity["label"]
                }
            )
            
            # Add entity-document relation
            self.graph.add_relation(
                subject_id=entity_id,
                predicate="mentioned_in",
                object_id=document_id,
                properties={
                    "start": entity["start"],
                    "end": entity["end"]
                }
            )
            
            # Add to cache
            self.entity_cache.add(entity_id)
        
        # Extract relationships
        relationships = self.extractor.extract_relationships(document_text)
        
        # Process each relationship
        for rel in relationships:
            subject = rel["subject"]
            object_entity = rel["object"]
            
            # Get entity IDs
            subject_id = self._generate_entity_id(subject["text"], subject["type"])
            object_id = self._generate_entity_id(object_entity["text"], object_entity["type"])
            
            # Add relation
            self.graph.add_relation(
                subject_id=subject_id,
                predicate=rel["predicate"],
                object_id=object_id,
                properties={
                    "sentence": rel["sentence"],
                    "source_document": document_id
                }
            )
    
    def query_graph(self, query: str) -> List[Dict[str, Any]]:
        """Query the knowledge graph with natural language.
        
        Args:
            query: Natural language query
            
        Returns:
            List of relevant entities and relationships
        """
        # Extract entities from query
        query_entities = self.extractor.extract_entities(query)
        
        results = []
        
        # For each entity in query, find related information
        for query_entity in query_entities:
            entity_id = self._generate_entity_id(query_entity["text"], query_entity["type"])
            
            # Get entity from graph
            entity = self.graph.get_entity(entity_id)
            
            # If entity exists, get its relationships
            if entity:
                relations = self.graph.get_relations(entity_id)
                
                results.append({
                    "entity": entity,
                    "relations": relations
                })
        
        return results
    
    def get_entities_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """Get all entities of a specific type.
        
        Args:
            entity_type: Type of entity to retrieve
            
        Returns:
            List of entities
        """
        return self.graph.query(query_entity_type=entity_type)
    
    def get_relationships_by_predicate(self, predicate: str) -> List[Dict[str, Any]]:
        """Get all relationships of a specific type.
        
        Args:
            predicate: Type of relationship to retrieve
            
        Returns:
            List of related entities
        """
        return self.graph.query(query_relation=predicate)
    
    def visualize_graph(self, output_path: Optional[str] = None):
        """Visualize the knowledge graph.
        
        Args:
            output_path: Path to save the visualization image
        """
        self.graph.visualize(output_path=output_path)
    
    def save_graph(self):
        """Save the knowledge graph to file."""
        self.graph.save(self.graph_path)
    
    def _generate_entity_id(self, entity_text: str, entity_type: str) -> str:
        """Generate a consistent ID for an entity.
        
        Args:
            entity_text: Text of the entity
            entity_type: Type of the entity
            
        Returns:
            Unique entity ID
        """
        # Normalize text
        normalized_text = entity_text.lower().strip()
        
        # Generate hash
        hash_obj = hashlib.md5(f"{normalized_text}_{entity_type}".encode())
        return f"{entity_type.lower()}_{hash_obj.hexdigest()[:12]}"
