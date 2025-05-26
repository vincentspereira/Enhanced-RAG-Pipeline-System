from typing import List, Dict, Any, Optional, Set, Tuple
import networkx as nx
from dataclasses import dataclass
import spacy

@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    id: str
    name: str
    type: str
    properties: Dict[str, Any]

@dataclass
class Relation:
    """Represents a relation between entities."""
    source_id: str
    target_id: str
    type: str
    properties: Dict[str, Any]

class KnowledgeGraph:
    """Knowledge graph implementation using NetworkX."""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.nlp = spacy.load("en_core_web_sm")
        
    def add_entity(self, entity: Entity):
        """Add an entity to the graph."""
        self.graph.add_node(
            entity.id,
            name=entity.name,
            type=entity.type,
            **entity.properties
        )
    
    def add_relation(self, relation: Relation):
        """Add a relation between entities."""
        self.graph.add_edge(
            relation.source_id,
            relation.target_id,
            type=relation.type,
            **relation.properties
        )
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        if entity_id not in self.graph.nodes:
            return None
            
        node = self.graph.nodes[entity_id]
        return Entity(
            id=entity_id,
            name=node["name"],
            type=node["type"],
            properties={k: v for k, v in node.items() 
                       if k not in ["name", "type"]}
        )
    
    def get_relations(self, entity_id: str) -> List[Relation]:
        """Get all relations for an entity."""
        relations = []
        
        # Outgoing relations
        for _, target, data in self.graph.out_edges(entity_id, data=True):
            relations.append(Relation(
                source_id=entity_id,
                target_id=target,
                type=data["type"],
                properties={k: v for k, v in data.items() if k != "type"}
            ))
            
        # Incoming relations
        for source, _, data in self.graph.in_edges(entity_id, data=True):
            relations.append(Relation(
                source_id=source,
                target_id=entity_id,
                type=data["type"],
                properties={k: v for k, v in data.items() if k != "type"}
            ))
            
        return relations
    
    def extract_entities_from_text(self, text: str) -> List[Entity]:
        """Extract entities from text using SpaCy."""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entity = Entity(
                id=f"{ent.label_}_{len(entities)}",
                name=ent.text,
                type=ent.label_,
                properties={"start": ent.start_char, "end": ent.end_char}
            )
            entities.append(entity)
            
        return entities
    
    def extract_relations_from_text(self, text: str) -> List[Relation]:
        """Extract relations from text using dependency parsing."""
        doc = self.nlp(text)
        relations = []
        
        for token in doc:
            if token.dep_ in ["nsubj", "dobj", "pobj"]:
                relation = Relation(
                    source_id=str(token.head.i),
                    target_id=str(token.i),
                    type=token.dep_,
                    properties={
                        "source_text": token.head.text,
                        "target_text": token.text
                    }
                )
                relations.append(relation)
                
        return relations
    
    def find_paths(self, source_id: str, target_id: str, 
                  max_length: int = 3) -> List[List[str]]:
        """Find all paths between two entities."""
        try:
            return list(nx.all_simple_paths(
                self.graph, source_id, target_id, cutoff=max_length
            ))
        except nx.NetworkXNoPath:
            return []
    
    def get_subgraph(self, entity_ids: Set[str], 
                     depth: int = 1) -> 'KnowledgeGraph':
        """Get a subgraph centered around given entities."""
        # Get nodes within n hops of the seed nodes
        nodes = set(entity_ids)
        for _ in range(depth):
            neighbors = set()
            for node in nodes:
                neighbors.update(self.graph.predecessors(node))
                neighbors.update(self.graph.successors(node))
            nodes.update(neighbors)
        
        # Create new graph with the selected nodes
        subgraph = KnowledgeGraph()
        subgraph.graph = self.graph.subgraph(nodes).copy()
        return subgraph
