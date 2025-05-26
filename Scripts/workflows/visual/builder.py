"""
Visual workflow builder for creating and managing RAG workflows.

This module provides functionality for creating, visualizing, and managing
RAG workflows through a visual interface.
"""
import os
import json
import logging
import uuid
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from pathlib import Path
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WorkflowNode:
    """Node in a workflow graph."""
    
    def __init__(
        self, 
        node_id: str,
        node_type: str,
        name: str,
        description: str = None,
        config: Dict[str, Any] = None,
        position: Dict[str, float] = None
    ):
        """Initialize a workflow node.
        
        Args:
            node_id: Unique node ID
            node_type: Node type
            name: Node name
            description: Optional node description
            config: Optional node configuration
            position: Optional position in the visual editor
        """
        self.node_id = node_id
        self.node_type = node_type
        self.name = name
        self.description = description
        self.config = config or {}
        self.position = position or {"x": 0, "y": 0}
        self.inputs = []
        self.outputs = []
    
    def to_dict(self):
        """Convert to dictionary representation."""
        return {
            "id": self.node_id,
            "type": self.node_type,
            "name": self.name,
            "description": self.description,
            "config": self.config,
            "position": self.position,
            "inputs": self.inputs,
            "outputs": self.outputs
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create from dictionary representation."""
        node = cls(
            node_id=data["id"],
            node_type=data["type"],
            name=data["name"],
            description=data.get("description"),
            config=data.get("config", {}),
            position=data.get("position", {"x": 0, "y": 0})
        )
        
        node.inputs = data.get("inputs", [])
        node.outputs = data.get("outputs", [])
        
        return node


class WorkflowEdge:
    """Edge connecting nodes in a workflow graph."""
    
    def __init__(
        self, 
        edge_id: str,
        source_id: str,
        target_id: str,
        source_handle: str = None,
        target_handle: str = None,
        label: str = None
    ):
        """Initialize a workflow edge.
        
        Args:
            edge_id: Unique edge ID
            source_id: Source node ID
            target_id: Target node ID
            source_handle: Optional source handle
            target_handle: Optional target handle
            label: Optional edge label
        """
        self.edge_id = edge_id
        self.source_id = source_id
        self.target_id = target_id
        self.source_handle = source_handle
        self.target_handle = target_handle
        self.label = label
    
    def to_dict(self):
        """Convert to dictionary representation."""
        return {
            "id": self.edge_id,
            "source": self.source_id,
            "target": self.target_id,
            "sourceHandle": self.source_handle,
            "targetHandle": self.target_handle,
            "label": self.label
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create from dictionary representation."""
        return cls(
            edge_id=data["id"],
            source_id=data["source"],
            target_id=data["target"],
            source_handle=data.get("sourceHandle"),
            target_handle=data.get("targetHandle"),
            label=data.get("label")
        )


class WorkflowDefinition:
    """Definition of a workflow."""
    
    def __init__(
        self, 
        workflow_id: str,
        name: str,
        description: str = None,
        version: str = "1.0",
        created_at: str = None,
        updated_at: str = None,
        nodes: List[WorkflowNode] = None,
        edges: List[WorkflowEdge] = None,
        metadata: Dict[str, Any] = None
    ):
        """Initialize a workflow definition.
        
        Args:
            workflow_id: Unique workflow ID
            name: Workflow name
            description: Optional workflow description
            version: Workflow version
            created_at: Creation timestamp
            updated_at: Last update timestamp
            nodes: List of workflow nodes
            edges: List of workflow edges
            metadata: Optional workflow metadata
        """
        self.workflow_id = workflow_id
        self.name = name
        self.description = description
        self.version = version
        self.created_at = created_at or datetime.now().isoformat()
        self.updated_at = updated_at or self.created_at
        self.nodes = nodes or []
        self.edges = edges or []
        self.metadata = metadata or {}
    
    def to_dict(self):
        """Convert to dictionary representation."""
        return {
            "id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create from dictionary representation."""
        workflow = cls(
            workflow_id=data["id"],
            name=data["name"],
            description=data.get("description"),
            version=data.get("version", "1.0"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            metadata=data.get("metadata", {})
        )
        
        # Create nodes
        for node_data in data.get("nodes", []):
            workflow.nodes.append(WorkflowNode.from_dict(node_data))
        
        # Create edges
        for edge_data in data.get("edges", []):
            workflow.edges.append(WorkflowEdge.from_dict(edge_data))
        
        return workflow
    
    def add_node(
        self, 
        node_type: str,
        name: str,
        description: str = None,
        config: Dict[str, Any] = None,
        position: Dict[str, float] = None
    ):
        """Add a node to the workflow.
        
        Args:
            node_type: Node type
            name: Node name
            description: Optional node description
            config: Optional node configuration
            position: Optional position in the visual editor
            
        Returns:
            Created node
        """
        node_id = f"node_{uuid.uuid4().hex[:8]}"
        
        node = WorkflowNode(
            node_id=node_id,
            node_type=node_type,
            name=name,
            description=description,
            config=config,
            position=position
        )
        
        self.nodes.append(node)
        self.updated_at = datetime.now().isoformat()
        
        return node
    
    def add_edge(
        self, 
        source_id: str,
        target_id: str,
        source_handle: str = None,
        target_handle: str = None,
        label: str = None
    ):
        """Add an edge to the workflow.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            source_handle: Optional source handle
            target_handle: Optional target handle
            label: Optional edge label
            
        Returns:
            Created edge
        """
        edge_id = f"edge_{uuid.uuid4().hex[:8]}"
        
        edge = WorkflowEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            source_handle=source_handle,
            target_handle=target_handle,
            label=label
        )
        
        self.edges.append(edge)
        self.updated_at = datetime.now().isoformat()
        
        return edge
    
    def remove_node(self, node_id: str):
        """Remove a node from the workflow.
        
        Args:
            node_id: Node ID
            
        Returns:
            True if node was removed, False otherwise
        """
        for i, node in enumerate(self.nodes):
            if node.node_id == node_id:
                self.nodes.pop(i)
                
                # Remove connected edges
                self.edges = [
                    edge for edge in self.edges
                    if edge.source_id != node_id and edge.target_id != node_id
                ]
                
                self.updated_at = datetime.now().isoformat()
                return True
        
        return False
    
    def remove_edge(self, edge_id: str):
        """Remove an edge from the workflow.
        
        Args:
            edge_id: Edge ID
            
        Returns:
            True if edge was removed, False otherwise
        """
        for i, edge in enumerate(self.edges):
            if edge.edge_id == edge_id:
                self.edges.pop(i)
                self.updated_at = datetime.now().isoformat()
                return True
        
        return False
    
    def get_node(self, node_id: str):
        """Get a node by ID.
        
        Args:
            node_id: Node ID
            
        Returns:
            Node if found, None otherwise
        """
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        
        return None
    
    def get_edge(self, edge_id: str):
        """Get an edge by ID.
        
        Args:
            edge_id: Edge ID
            
        Returns:
            Edge if found, None otherwise
        """
        for edge in self.edges:
            if edge.edge_id == edge_id:
                return edge
        
        return None
    
    def update_node(
        self, 
        node_id: str,
        name: str = None,
        description: str = None,
        config: Dict[str, Any] = None,
        position: Dict[str, float] = None
    ):
        """Update a node in the workflow.
        
        Args:
            node_id: Node ID
            name: Optional new name
            description: Optional new description
            config: Optional new configuration
            position: Optional new position
            
        Returns:
            Updated node if found, None otherwise
        """
        node = self.get_node(node_id)
        if not node:
            return None
        
        if name is not None:
            node.name = name
        
        if description is not None:
            node.description = description
        
        if config is not None:
            node.config = config
        
        if position is not None:
            node.position = position
        
        self.updated_at = datetime.now().isoformat()
        
        return node


class WorkflowTemplateLibrary:
    """Library of workflow templates."""
    
    def __init__(self, templates_dir: str = "data/workflow_templates"):
        """Initialize the template library.
        
        Args:
            templates_dir: Directory for storing templates
        """
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
    
    def save_template(self, workflow: WorkflowDefinition, category: str = None):
        """Save a workflow as a template.
        
        Args:
            workflow: Workflow definition
            category: Optional template category
            
        Returns:
            Path to the saved template
        """
        # Create template data
        template_data = workflow.to_dict()
        
        if category:
            template_data["metadata"]["category"] = category
        
        # Create category directory if needed
        category_dir = self.templates_dir
        if category:
            category_dir = self.templates_dir / category
            category_dir.mkdir(exist_ok=True)
        
        # Save template
        template_path = category_dir / f"{workflow.workflow_id}.json"
        
        with open(template_path, "w") as f:
            json.dump(template_data, f, indent=2)
        
        logger.info(f"Saved workflow template: {template_path}")
        return template_path
    
    def load_template(self, template_id: str, category: str = None):
        """Load a workflow template.
        
        Args:
            template_id: Template ID
            category: Optional template category
            
        Returns:
            Loaded workflow definition
        """
        # Find template file
        if category:
            template_path = self.templates_dir / category / f"{template_id}.json"
        else:
            template_path = self.templates_dir / f"{template_id}.json"
            
            # Try to find in any category
            if not template_path.exists():
                for category_dir in self.templates_dir.glob("*"):
                    if category_dir.is_dir():
                        candidate_path = category_dir / f"{template_id}.json"
                        if candidate_path.exists():
                            template_path = candidate_path
                            break
        
        if not template_path.exists():
            raise ValueError(f"Template {template_id} not found")
        
        # Load template
        with open(template_path, "r") as f:
            template_data = json.load(f)
        
        return WorkflowDefinition.from_dict(template_data)
    
    def list_templates(self, category: str = None):
        """List available templates.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of template IDs
        """
        templates = []
        
        if category:
            # List templates in specific category
            category_dir = self.templates_dir / category
            if category_dir.exists():
                for template_file in category_dir.glob("*.json"):
                    templates.append({"id": template_file.stem, "category": category})
        else:
            # List all templates
            for template_file in self.templates_dir.glob("*.json"):
                templates.append({"id": template_file.stem, "category": None})
            
            # List templates in categories
            for category_dir in self.templates_dir.glob("*"):
                if category_dir.is_dir():
                    category = category_dir.name
                    for template_file in category_dir.glob("*.json"):
                        templates.append({"id": template_file.stem, "category": category})
        
        return templates
    
    def get_template_info(self, template_id: str, category: str = None):
        """Get information about a template.
        
        Args:
            template_id: Template ID
            category: Optional template category
            
        Returns:
            Template information
        """
        # Load template
        workflow = self.load_template(template_id, category)
        
        # Extract basic info
        return {
            "id": workflow.workflow_id,
            "name": workflow.name,
            "description": workflow.description,
            "version": workflow.version,
            "category": category or workflow.metadata.get("category"),
            "node_count": len(workflow.nodes),
            "created_at": workflow.created_at,
            "updated_at": workflow.updated_at
        }
    
    def delete_template(self, template_id: str, category: str = None):
        """Delete a template.
        
        Args:
            template_id: Template ID
            category: Optional template category
            
        Returns:
            True if template was deleted, False otherwise
        """
        # Find template file
        if category:
            template_path = self.templates_dir / category / f"{template_id}.json"
        else:
            template_path = self.templates_dir / f"{template_id}.json"
            
            # Try to find in any category
            if not template_path.exists():
                for category_dir in self.templates_dir.glob("*"):
                    if category_dir.is_dir():
                        candidate_path = category_dir / f"{template_id}.json"
                        if candidate_path.exists():
                            template_path = candidate_path
                            break
        
        if not template_path.exists():
            return False
        
        # Delete template
        template_path.unlink()
        logger.info(f"Deleted workflow template: {template_path}")
        
        return True
    
    def create_template_from_scratch(
        self, 
        name: str,
        description: str = None,
        category: str = None
    ):
        """Create a new template from scratch.
        
        Args:
            name: Template name
            description: Optional template description
            category: Optional template category
            
        Returns:
            Created workflow definition
        """
        workflow_id = f"template_{uuid.uuid4().hex[:8]}"
        
        workflow = WorkflowDefinition(
            workflow_id=workflow_id,
            name=name,
            description=description,
            metadata={"category": category} if category else {}
        )
        
        # Save template
        self.save_template(workflow, category)
        
        return workflow


class WorkflowStorage:
    """Storage for workflow definitions."""
    
    def __init__(self, workflows_dir: str = "data/workflows"):
        """Initialize the workflow storage.
        
        Args:
            workflows_dir: Directory for storing workflows
        """
        self.workflows_dir = Path(workflows_dir)
        self.workflows_dir.mkdir(parents=True, exist_ok=True)
    
    def save_workflow(self, workflow: WorkflowDefinition):
        """Save a workflow.
        
        Args:
            workflow: Workflow definition
            
        Returns:
            Path to the saved workflow
        """
        # Update timestamp
        workflow.updated_at = datetime.now().isoformat()
        
        # Create workflow data
        workflow_data = workflow.to_dict()
        
        # Save workflow
        workflow_path = self.workflows_dir / f"{workflow.workflow_id}.json"
        
        with open(workflow_path, "w") as f:
            json.dump(workflow_data, f, indent=2)
        
        logger.info(f"Saved workflow: {workflow_path}")
        return workflow_path
    
    def load_workflow(self, workflow_id: str):
        """Load a workflow.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            Loaded workflow definition
        """
        workflow_path = self.workflows_dir / f"{workflow_id}.json"
        
        if not workflow_path.exists():
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Load workflow
        with open(workflow_path, "r") as f:
            workflow_data = json.load(f)
        
        return WorkflowDefinition.from_dict(workflow_data)
    
    def list_workflows(self):
        """List available workflows.
        
        Returns:
            List of workflow IDs
        """
        workflows = []
        
        for workflow_file in self.workflows_dir.glob("*.json"):
            workflows.append(workflow_file.stem)
        
        return workflows
    
    def get_workflow_info(self, workflow_id: str):
        """Get information about a workflow.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            Workflow information
        """
        # Load workflow
        workflow = self.load_workflow(workflow_id)
        
        # Extract basic info
        return {
            "id": workflow.workflow_id,
            "name": workflow.name,
            "description": workflow.description,
            "version": workflow.version,
            "node_count": len(workflow.nodes),
            "created_at": workflow.created_at,
            "updated_at": workflow.updated_at
        }
    
    def delete_workflow(self, workflow_id: str):
        """Delete a workflow.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            True if workflow was deleted, False otherwise
        """
        workflow_path = self.workflows_dir / f"{workflow_id}.json"
        
        if not workflow_path.exists():
            return False
        
        # Delete workflow
        workflow_path.unlink()
        logger.info(f"Deleted workflow: {workflow_path}")
        
        return True
    
    def create_workflow_from_template(
        self, 
        template_id: str,
        name: str = None,
        description: str = None,
        category: str = None
    ):
        """Create a workflow from a template.
        
        Args:
            template_id: Template ID
            name: Optional workflow name
            description: Optional workflow description
            category: Optional template category
            
        Returns:
            Created workflow definition
        """
        # Load template
        template_library = WorkflowTemplateLibrary()
        template = template_library.load_template(template_id, category)
        
        # Create new workflow with unique ID
        workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
        
        workflow = WorkflowDefinition(
            workflow_id=workflow_id,
            name=name or template.name,
            description=description or template.description,
            nodes=template.nodes,
            edges=template.edges,
            metadata=template.metadata
        )
        
        # Save workflow
        self.save_workflow(workflow)
        
        return workflow
    
    def create_workflow_from_scratch(
        self, 
        name: str,
        description: str = None
    ):
        """Create a new workflow from scratch.
        
        Args:
            name: Workflow name
            description: Optional workflow description
            
        Returns:
            Created workflow definition
        """
        workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
        
        workflow = WorkflowDefinition(
            workflow_id=workflow_id,
            name=name,
            description=description
        )
        
        # Save workflow
        self.save_workflow(workflow)
        
        return workflow


class NodeTypeLibrary:
    """Library of available node types for workflows."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the node type library.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.node_types = self._load_node_types()
    
    def _load_node_types(self):
        """Load node types from configuration.
        
        Returns:
            Dictionary of node types
        """
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            
            # Extract workflow node types
            node_types = config.get("workflows", {}).get("node_types", {})
            
            # If no node types defined, create default types
            if not node_types:
                node_types = self._create_default_node_types()
                
                # Update config file
                if "workflows" not in config:
                    config["workflows"] = {}
                
                config["workflows"]["node_types"] = node_types
                
                with open(self.config_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False)
            
            return node_types
        
        except Exception as e:
            logger.warning(f"Failed to load node types: {e}")
            return self._create_default_node_types()
    
    def _create_default_node_types(self):
        """Create default node types.
        
        Returns:
            Dictionary of default node types
        """
        return {
            "document_source": {
                "name": "Document Source",
                "description": "Source of documents for the RAG pipeline",
                "category": "input",
                "inputs": [],
                "outputs": ["documents"],
                "config_schema": {
                    "source_type": {
                        "type": "string",
                        "enum": ["file", "folder", "database", "api"],
                        "default": "file"
                    },
                    "path": {
                        "type": "string"
                    }
                }
            },
            "text_splitter": {
                "name": "Text Splitter",
                "description": "Splits documents into chunks",
                "category": "processing",
                "inputs": ["documents"],
                "outputs": ["chunks"],
                "config_schema": {
                    "chunk_size": {
                        "type": "number",
                        "default": 512
                    },
                    "chunk_overlap": {
                        "type": "number",
                        "default": 50
                    }
                }
            },
            "embedding_generator": {
                "name": "Embedding Generator",
                "description": "Generates embeddings for text chunks",
                "category": "processing",
                "inputs": ["chunks"],
                "outputs": ["embeddings"],
                "config_schema": {
                    "model": {
                        "type": "string",
                        "default": "Snowflake-Labs/arctic-embed2"
                    },
                    "batch_size": {
                        "type": "number",
                        "default": 32
                    }
                }
            },
            "vector_store": {
                "name": "Vector Store",
                "description": "Stores embeddings in a vector database",
                "category": "storage",
                "inputs": ["embeddings"],
                "outputs": ["stored_vectors"],
                "config_schema": {
                    "provider": {
                        "type": "string",
                        "enum": ["qdrant", "weaviate", "pinecone", "milvus"],
                        "default": "qdrant"
                    },
                    "collection_name": {
                        "type": "string",
                        "default": "documents"
                    }
                }
            },
            "query_processor": {
                "name": "Query Processor",
                "description": "Processes user queries",
                "category": "processing",
                "inputs": ["query"],
                "outputs": ["processed_query"],
                "config_schema": {
                    "rewrite_query": {
                        "type": "boolean",
                        "default": False
                    },
                    "use_hybrid": {
                        "type": "boolean",
                        "default": True
                    }
                }
            },
            "retriever": {
                "name": "Retriever",
                "description": "Retrieves relevant documents from vector store",
                "category": "processing",
                "inputs": ["processed_query", "stored_vectors"],
                "outputs": ["retrieved_documents"],
                "config_schema": {
                    "top_k": {
                        "type": "number",
                        "default": 5
                    },
                    "threshold": {
                        "type": "number",
                        "default": 0.7
                    }
                }
            },
            "llm": {
                "name": "Language Model",
                "description": "Generates responses using an LLM",
                "category": "generation",
                "inputs": ["processed_query", "retrieved_documents"],
                "outputs": ["response"],
                "config_schema": {
                    "provider": {
                        "type": "string",
                        "enum": ["openai", "anthropic", "huggingface", "local"],
                        "default": "huggingface"
                    },
                    "model": {
                        "type": "string",
                        "default": "meta-llama/Llama-2-7b-hf"
                    }
                }
            },
            "output_formatter": {
                "name": "Output Formatter",
                "description": "Formats the output response",
                "category": "output",
                "inputs": ["response"],
                "outputs": ["formatted_response"],
                "config_schema": {
                    "format": {
                        "type": "string",
                        "enum": ["text", "json", "html", "markdown"],
                        "default": "text"
                    }
                }
            }
        }
    
    def get_node_type(self, node_type: str):
        """Get a node type by name.
        
        Args:
            node_type: Node type name
            
        Returns:
            Node type definition if found, None otherwise
        """
        return self.node_types.get(node_type)
    
    def list_node_types(self, category: str = None):
        """List available node types.
        
        Args:
            category: Optional category filter
            
        Returns:
            Dictionary of node types
        """
        if category:
            return {
                name: definition
                for name, definition in self.node_types.items()
                if definition.get("category") == category
            }
        else:
            return self.node_types
    
    def register_node_type(
        self, 
        type_name: str,
        name: str,
        description: str,
        category: str,
        inputs: List[str],
        outputs: List[str],
        config_schema: Dict[str, Any]
    ):
        """Register a new node type.
        
        Args:
            type_name: Node type name
            name: Display name
            description: Node type description
            category: Node category
            inputs: List of input types
            outputs: List of output types
            config_schema: Configuration schema
            
        Returns:
            True if node type was registered, False otherwise
        """
        # Create node type definition
        node_type = {
            "name": name,
            "description": description,
            "category": category,
            "inputs": inputs,
            "outputs": outputs,
            "config_schema": config_schema
        }
        
        # Add to node types
        self.node_types[type_name] = node_type
        
        # Save updated configuration
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            
            if "workflows" not in config:
                config["workflows"] = {}
            
            config["workflows"]["node_types"] = self.node_types
            
            with open(self.config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to save node type: {e}")
            return False
