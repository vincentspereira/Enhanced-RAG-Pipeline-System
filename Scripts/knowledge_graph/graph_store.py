"""
Knowledge graph storage and query module.
"""
from typing import List, Dict, Any, Tuple, Set, Optional
import networkx as nx
import matplotlib.pyplot as plt
import json
import os

class KnowledgeGraph:
    """Knowledge graph implementation using NetworkX."""
    
    def __init__(self, name: str = "knowledge_graph"):
        """Initialize a knowledge graph.
        
        Args:
            name: Name of the knowledge graph
        """
        self.name = name
        self.graph = nx.DiGraph()
        
    def add_entity(self, entity_id: str, entity_type: str, properties: Dict[str, Any]):
        """Add an entity to the knowledge graph.
        
        Args:
            entity_id: Unique identifier for the entity
            entity_type: Type of the entity (e.g., Person, Organization)
            properties: Additional properties of the entity
        """
        # Add entity as a node
        self.graph.add_node(
            entity_id,
            type=entity_type,
            **properties
        )
        
    def add_relation(self, subject_id: str, predicate: str, object_id: str, 
                    properties: Optional[Dict[str, Any]] = None):
        """Add a relationship between entities.
        
        Args:
            subject_id: ID of the subject entity
            predicate: Type of relationship
            object_id: ID of the object entity
            properties: Additional properties of the relationship
        """
        # Add relationship as an edge
        if properties is None:
            properties = {}
            
        self.graph.add_edge(
            subject_id,
            object_id,
            predicate=predicate,
            **properties
        )
        
    def get_entity(self, entity_id: str) -> Dict[str, Any]:
        """Get an entity by ID.
        
        Args:
            entity_id: ID of the entity to retrieve
            
        Returns:
            Entity data including properties
        """
        if entity_id not in self.graph.nodes:
            return None
            
        node_data = self.graph.nodes[entity_id]
        return {
            "id": entity_id,
            **node_data
        }
        
    def get_relations(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get all relationships involving an entity.
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            List of relationships where the entity is subject or object
        """
        relations = []
        
        # Get outgoing edges (entity is subject)
        for subject, object_id, edge_data in self.graph.out_edges(entity_id, data=True):
            relations.append({
                "subject": subject,
                "predicate": edge_data.get("predicate", "unknown"),
                "object": object_id,
                "direction": "outgoing",
                **{k: v for k, v in edge_data.items() if k != "predicate"}
            })
            
        # Get incoming edges (entity is object)
        for subject_id, object_id, edge_data in self.graph.in_edges(entity_id, data=True):
            relations.append({
                "subject": subject_id,
                "predicate": edge_data.get("predicate", "unknown"),
                "object": object_id,
                "direction": "incoming",
                **{k: v for k, v in edge_data.items() if k != "predicate"}
            })
            
        return relations
        
    def query(self, query_entity_type: Optional[str] = None, 
             query_relation: Optional[str] = None,
             query_properties: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Query the knowledge graph.
        
        Args:
            query_entity_type: Optional filter by entity type
            query_relation: Optional filter by relation type
            query_properties: Optional filter by entity properties
            
        Returns:
            List of matching entities
        """
        results = []
        
        # Filter nodes by type and properties
        for node_id, node_data in self.graph.nodes(data=True):
            # Check entity type
            if query_entity_type and node_data.get("type") != query_entity_type:
                continue
                
            # Check properties
            if query_properties:
                match = True
                for key, value in query_properties.items():
                    if key not in node_data or node_data[key] != value:
                        match = False
                        break
                if not match:
                    continue
            
            # If relation filter specified, check relations
            if query_relation:
                # Check if node has any edges with this relation
                has_relation = False
                
                # Check outgoing edges
                for _, _, edge_data in self.graph.out_edges(node_id, data=True):
                    if edge_data.get("predicate") == query_relation:
                        has_relation = True
                        break
                        
                # Check incoming edges
                if not has_relation:
                    for _, _, edge_data in self.graph.in_edges(node_id, data=True):
                        if edge_data.get("predicate") == query_relation:
                            has_relation = True
                            break
                            
                if not has_relation:
                    continue
            
            # Add to results
            results.append({
                "id": node_id,
                **node_data
            })
        
        return results
    
    def visualize(self, output_path: Optional[str] = None, 
                 max_nodes: int = 100):
        """Visualize the knowledge graph.
        
        Args:
            output_path: Path to save the visualization image
            max_nodes: Maximum number of nodes to include in visualization
        """
        if len(self.graph) > max_nodes:
            # Create a subgraph with limited nodes
            important_nodes = list(self.graph.nodes())[:max_nodes]
            subgraph = self.graph.subgraph(important_nodes)
        else:
            subgraph = self.graph
            
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Create position layout
        pos = nx.spring_layout(subgraph)
        
        # Draw nodes
        node_colors = []
        for node in subgraph.nodes():
            node_type = subgraph.nodes[node].get("type", "unknown")
            # Map types to colors
            if node_type == "Person":
                node_colors.append("lightblue")
            elif node_type == "Organization":
                node_colors.append("lightgreen")
            elif node_type == "Location":
                node_colors.append("lightcoral")
            else:
                node_colors.append("gray")
                
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=500, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(subgraph, pos, width=1.0, alpha=0.5, arrows=True)
        
        # Draw labels
        nx.draw_networkx_labels(subgraph, pos, font_size=8)
        
        # Add edge labels (predicates)
        edge_labels = {(u, v): data.get("predicate", "") 
                      for u, v, data in subgraph.edges(data=True)}
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=7)
        
        plt.title(f"Knowledge Graph: {self.name} (showing {len(subgraph)} of {len(self.graph)} nodes)")
        plt.axis("off")
        
        # Save or show
        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
        else:
            plt.show()
            
        plt.close()
    
    def save(self, file_path: str):
        """Save the knowledge graph to a file.
        
        Args:
            file_path: Path to save the graph
        """
        # Convert graph to dict for JSON serialization
        data = {
            "name": self.name,
            "nodes": [],
            "edges": []
        }
        
        # Add nodes
        for node_id, node_data in self.graph.nodes(data=True):
            data["nodes"].append({
                "id": node_id,
                **node_data
            })
            
        # Add edges
        for source, target, edge_data in self.graph.edges(data=True):
            data["edges"].append({
                "source": source,
                "target": target,
                **edge_data
            })
            
        # Save to file
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, file_path: str) -> "KnowledgeGraph":
        """Load a knowledge graph from a file.
        
        Args:
            file_path: Path to load the graph from
            
        Returns:
            Loaded knowledge graph
        """
        with open(file_path, "r") as f:
            data = json.load(f)
            
        # Create graph
        graph = cls(name=data.get("name", "knowledge_graph"))
        
        # Add nodes
        for node in data.get("nodes", []):
            node_id = node.pop("id")
            node_type = node.pop("type", "unknown")
            graph.add_entity(node_id, node_type, node)
            
        # Add edges
        for edge in data.get("edges", []):
            source = edge.pop("source")
            target = edge.pop("target")
            predicate = edge.pop("predicate", "related_to")
            graph.add_relation(source, predicate, target, edge)
            
        return graph
