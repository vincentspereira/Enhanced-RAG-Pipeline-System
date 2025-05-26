"""
Initialization module for knowledge graph components.
"""
from typing import Dict, Any, Optional, Tuple
import os

from Scripts.knowledge_graph.builder import KnowledgeGraphBuilder
from Scripts.knowledge_graph.visualization import KnowledgeGraphVisualizer

def init_knowledge_graph(config: Dict[str, Any]) -> Tuple[Optional[KnowledgeGraphBuilder], Optional[KnowledgeGraphVisualizer]]:
    """Initialize knowledge graph from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (knowledge graph builder, visualizer) or (None, None) if disabled
    """
    # Check if knowledge graph is enabled
    kg_config = config.get("knowledge_graph", {})
    enabled = kg_config.get("enabled", False)
    
    if not enabled:
        return None, None
        
    # Get configuration
    graph_name = kg_config.get("name", "knowledge_graph")
    graph_path = kg_config.get("path")
    
    if not graph_path:
        storage_dir = kg_config.get("storage_dir", "data/knowledge_graphs")
        os.makedirs(storage_dir, exist_ok=True)
        graph_path = f"{storage_dir}/{graph_name}.json"
    
    model_name = kg_config.get("model", "en_core_web_lg")
    
    # Initialize builder
    builder = KnowledgeGraphBuilder(
        graph_name=graph_name,
        graph_path=graph_path,
        model_name=model_name
    )
    
    # Initialize visualizer
    visualization_dir = kg_config.get("visualization_dir", "data/visualizations")
    visualizer = KnowledgeGraphVisualizer(output_dir=visualization_dir)
    
    return builder, visualizer
