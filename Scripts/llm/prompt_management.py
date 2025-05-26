"""
Prompt management system for LLM services.

This module provides functionality for managing, versioning, and optimizing prompts
used with LLM services in the RAG pipeline.
"""
from typing import Dict, Any, List, Optional, Union
import os
import json
import yaml
import logging
import uuid
from datetime import datetime
import re
import numpy as np
from collections import defaultdict

# Import default templates
from .default_prompts import DefaultPromptTemplates

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PromptTemplate:
    """Template for structured prompts with variable interpolation."""
    
    def __init__(self, template: str, name: str = None, description: str = None, 
                variables: Optional[List[str]] = None, version: str = "1.0"):
        """Initialize a prompt template.
        
        Args:
            template: The prompt template string with variable placeholders
            name: Optional name for the template
            description: Optional description
            variables: Optional list of variable names (auto-detected if not provided)
            version: Version of the template
        """
        self.template = template
        self.name = name or f"template_{uuid.uuid4().hex[:8]}"
        self.description = description or ""
        self.version = version
        
        # Auto-detect variables if not provided
        if variables is None:
            self.variables = self._detect_variables(template)
        else:
            self.variables = variables
        
        # Validate template
        self._validate_template()
    
    def _detect_variables(self, template: str) -> List[str]:
        """Detect variable placeholders in the template.
        
        Args:
            template: The prompt template string
            
        Returns:
            List of variable names
        """
        # Find all {variable} patterns
        variables = re.findall(r'\{([^{}]+)\}', template)
        return list(set(variables))
    
    def _validate_template(self):
        """Validate that the template has all required variables."""
        template_vars = self._detect_variables(self.template)
        for var in template_vars:
            if var not in self.variables:
                self.variables.append(var)
    
    def format(self, **kwargs) -> str:
        """Format the template with provided variable values.
        
        Args:
            **kwargs: Variable values for substitution
            
        Returns:
            Formatted prompt string
            
        Raises:
            ValueError: If required variables are missing
        """
        # Check for missing variables
        missing_vars = [var for var in self.variables if var not in kwargs]
        if missing_vars:
            raise ValueError(f"Missing required variables: {', '.join(missing_vars)}")
        
        # Format template
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Error formatting template: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary representation.
        
        Returns:
            Dictionary representation of the template
        """
        return {
            "name": self.name,
            "description": self.description,
            "template": self.template,
            "variables": self.variables,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptTemplate':
        """Create template from dictionary representation.
        
        Args:
            data: Dictionary representation of the template
            
        Returns:
            New PromptTemplate instance
        """
        return cls(
            template=data["template"],
            name=data.get("name"),
            description=data.get("description", ""),
            variables=data.get("variables"),
            version=data.get("version", "1.0")
        )

class PromptVersion:
    """Class representing a specific version of a prompt template."""
    
    def __init__(
        self, 
        version: str,
        template: str,
        metrics: Optional[Dict[str, Any]] = None,
        created_at: Optional[str] = None
    ):
        """Initialize a prompt version.
        
        Args:
            version: Version string (e.g., '1.0')
            template: Prompt template text
            metrics: Optional performance metrics
            created_at: Creation timestamp
        """
        self.version = version
        self.template = template
        self.metrics = metrics or {}
        self.created_at = created_at or datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "version": self.version,
            "template": self.template,
            "metrics": self.metrics,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptVersion':
        """Create from dictionary representation."""
        return cls(
            version=data["version"],
            template=data["template"],
            metrics=data.get("metrics", {}),
            created_at=data.get("created_at")
        )


class PromptManager:
    """Manager for prompt templates and versions."""
    
    def __init__(self, library: PromptLibrary):
        """Initialize the prompt manager.
        
        Args:
            library: PromptLibrary instance
        """
        self.library = library
        self.versions_dir = os.path.join(library.library_path, "versions")
        self.feedback_dir = os.path.join(library.library_path, "feedback")
        
        # Create directories
        os.makedirs(self.versions_dir, exist_ok=True)
        os.makedirs(self.feedback_dir, exist_ok=True)
        
        # Map of template name to versions
        self.template_versions: Dict[str, List[PromptVersion]] = {}
        
        # Load existing versions
        self._load_template_versions()
    
    def _load_template_versions(self):
        """Load template versions from the versions directory."""
        # Get all template names
        for template_name in self.library.templates.keys():
            # Check for version directory
            version_dir = os.path.join(self.versions_dir, template_name)
            if os.path.exists(version_dir) and os.path.isdir(version_dir):
                versions = []
                
                # Load each version file
                for filename in os.listdir(version_dir):
                    if filename.endswith('.json'):
                        try:
                            filepath = os.path.join(version_dir, filename)
                            with open(filepath, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                versions.append(PromptVersion.from_dict(data))
                        except Exception as e:
                            logger.warning(f"Error loading version from {filepath}: {e}")
                
                # Store versions
                if versions:
                    self.template_versions[template_name] = versions
    
    def add_template_version(self, template_name: str, version: PromptVersion) -> bool:
        """Add a new version of a template.
        
        Args:
            template_name: Name of the template
            version: PromptVersion to add
            
        Returns:
            True if added successfully, False otherwise
        """
        # Check if template exists
        if not self.library.get_template(template_name):
            logger.error(f"Cannot add version: Template '{template_name}' not found")
            return False
        
        try:
            # Create version directory if needed
            version_dir = os.path.join(self.versions_dir, template_name)
            os.makedirs(version_dir, exist_ok=True)
            
            # Add to memory
            if template_name not in self.template_versions:
                self.template_versions[template_name] = []
            self.template_versions[template_name].append(version)
            
            # Save to file
            filepath = os.path.join(version_dir, f"{version.version}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(version.to_dict(), f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error adding version for {template_name}: {e}")
            return False
    
    def get_template_versions(self, template_name: str) -> List[PromptVersion]:
        """Get all versions of a template.
        
        Args:
            template_name: Name of the template
            
        Returns:
            List of PromptVersion instances
        """
        return self.template_versions.get(template_name, [])
    
    def get_template_version(self, template_name: str, version: str) -> Optional[PromptVersion]:
        """Get a specific version of a template.
        
        Args:
            template_name: Name of the template
            version: Version string
            
        Returns:
            PromptVersion or None if not found
        """
        versions = self.get_template_versions(template_name)
        for v in versions:
            if v.version == version:
                return v
        return None
    
    def update_version_metrics(self, template_name: str, version: str, 
                             metrics: Dict[str, Any]) -> bool:
        """Update metrics for a template version.
        
        Args:
            template_name: Name of the template
            version: Version string
            metrics: Metrics to update
            
        Returns:
            True if updated successfully, False otherwise
        """
        # Get version
        prompt_version = self.get_template_version(template_name, version)
        if not prompt_version:
            return False
        
        try:
            # Update metrics
            for key, value in metrics.items():
                prompt_version.metrics[key] = value
            
            # Save to file
            version_dir = os.path.join(self.versions_dir, template_name)
            filepath = os.path.join(version_dir, f"{version}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(prompt_version.to_dict(), f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error updating metrics for {template_name} v{version}: {e}")
            return False
    
    def record_feedback(self, template_name: str, version: str, 
                       feedback_data: Dict[str, Any], user_id: Optional[str] = None) -> bool:
        """Record feedback for a template version.
        
        Args:
            template_name: Name of the template
            version: Version string
            feedback_data: Feedback data
            user_id: Optional user ID
            
        Returns:
            True if recorded successfully, False otherwise
        """
        try:
            # Create feedback entry
            feedback_entry = {
                "template_name": template_name,
                "version": version,
                "user_id": user_id,
                "feedback": feedback_data,
                "timestamp": datetime.now().isoformat(),
                "id": uuid.uuid4().hex
            }
            
            # Create feedback directory if needed
            template_feedback_dir = os.path.join(self.feedback_dir, template_name)
            os.makedirs(template_feedback_dir, exist_ok=True)
            
            # Save to file
            filepath = os.path.join(template_feedback_dir, f"{feedback_entry['id']}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(feedback_entry, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error recording feedback for {template_name} v{version}: {e}")
            return False
    
    def get_feedback(self, template_name: str, version: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get feedback for a template.
        
        Args:
            template_name: Name of the template
            version: Optional version string to filter by
            
        Returns:
            List of feedback entries
        """
        feedback = []
        
        # Check for feedback directory
        template_feedback_dir = os.path.join(self.feedback_dir, template_name)
        if not os.path.exists(template_feedback_dir):
            return feedback
        
        # Load feedback entries
        for filename in os.listdir(template_feedback_dir):
            if filename.endswith('.json'):
                try:
                    filepath = os.path.join(template_feedback_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        entry = json.load(f)
                        
                        # Filter by version if specified
                        if version is None or entry.get("version") == version:
                            feedback.append(entry)
                except Exception as e:
                    logger.warning(f"Error loading feedback from {filepath}: {e}")
        
        return feedback
    

class PromptOptimizer:
    """Optimizer for improving prompt effectiveness."""
    
    def __init__(self, library: PromptLibrary):
        """Initialize the prompt optimizer.
        
        Args:
            library: PromptLibrary instance
        """
        self.library = library
        self.results_dir = os.path.join(library.library_path, "optimization_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Track optimization results
        self.optimization_results = {}
    
    def track_prompt_performance(self, template_name: str, prompt: str, 
                                variables: Dict[str, Any], metrics: Dict[str, float],
                                llm_name: Optional[str] = None) -> str:
        """Track performance of a prompt.
        
        Args:
            template_name: Name of the template used
            prompt: The formatted prompt text
            variables: Variables used in the prompt
            metrics: Performance metrics (e.g., relevance, accuracy)
            llm_name: Optional name of the LLM used
            
        Returns:
            ID of the tracked result
        """
        # Create unique ID for this result
        result_id = uuid.uuid4().hex
        
        # Create result entry
        result = {
            "template_name": template_name,
            "prompt": prompt,
            "variables": variables,
            "metrics": metrics,
            "llm_name": llm_name,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store result
        result_path = os.path.join(self.results_dir, f"{result_id}.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        
        # Update in-memory results
        if template_name not in self.optimization_results:
            self.optimization_results[template_name] = []
        
        self.optimization_results[template_name].append({
            "id": result_id,
            "metrics": metrics,
            "timestamp": result["timestamp"]
        })
        
        return result_id
    
    def analyze_template_performance(self, template_name: str) -> Dict[str, Any]:
        """Analyze performance of a template.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Analysis results
        """
        # Load results if not already in memory
        if template_name not in self.optimization_results:
            self._load_template_results(template_name)
        
        results = self.optimization_results.get(template_name, [])
        
        if not results:
            return {
                "template_name": template_name,
                "results_count": 0,
                "message": "No performance data available"
            }
        
        # Get all metric names
        metric_names = set()
        for result in results:
            metric_names.update(result["metrics"].keys())
        
        # Calculate statistics for each metric
        metrics_stats = {}
        for metric in metric_names:
            values = [r["metrics"].get(metric, 0) for r in results if metric in r["metrics"]]
            if values:
                metrics_stats[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
        
        # Sort results by timestamp
        sorted_results = sorted(results, key=lambda x: x["timestamp"])
        
        # Detect trends
        trends = {}
        if len(sorted_results) >= 3:
            for metric in metric_names:
                values = [r["metrics"].get(metric, 0) for r in sorted_results if metric in r["metrics"]]
                if len(values) >= 3:
                    # Simple trend: positive if last > first, negative otherwise
                    trend = "positive" if values[-1] > values[0] else "negative"
                    trends[metric] = {
                        "direction": trend,
                        "change": values[-1] - values[0]
                    }
        
        return {
            "template_name": template_name,
            "results_count": len(results),
            "metrics_stats": metrics_stats,
            "trends": trends,
            "latest_timestamp": sorted_results[-1]["timestamp"] if sorted_results else None
        }
    
    def suggest_improvements(self, template_name: str) -> Dict[str, Any]:
        """Suggest improvements for a template.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Improvement suggestions
        """
        # Get template
        template = self.library.get_template(template_name)
        if not template:
            return {"error": f"Template '{template_name}' not found"}
        
        # Analyze performance
        analysis = self.analyze_template_performance(template_name)
        
        # Basic suggestions
        suggestions = []
        
        # Check template length
        if len(template.template) < 50:
            suggestions.append({
                "type": "length",
                "message": "Template is very short. Consider adding more context or instructions."
            })
        elif len(template.template) > 1000:
            suggestions.append({
                "type": "length",
                "message": "Template is very long. Consider simplifying or breaking into separate templates."
            })
        
        # Check for common elements
        if "context" not in template.variables:
            suggestions.append({
                "type": "variable",
                "message": "Consider adding a 'context' variable to provide background information."
            })
        
        if "instructions" not in template.variables:
            suggestions.append({
                "type": "variable",
                "message": "Consider adding an 'instructions' variable for clear guidance."
            })
        
        # Check for patterns
        if not re.search(r'\b(respond|answer|write|generate|create)\b', template.template, re.IGNORECASE):
            suggestions.append({
                "type": "directive",
                "message": "Add clear directive verbs (respond, answer, write, etc.) to guide the model."
            })
        
        # Check for formatting guidance
        if not re.search(r'(format|structure|organize|layout)', template.template, re.IGNORECASE):
            suggestions.append({
                "type": "format",
                "message": "Consider adding formatting instructions for structured outputs."
            })
        
        # Performance-based suggestions
        if "metrics_stats" in analysis:
            metrics = analysis["metrics_stats"]
            
            # If any metric is low
            for metric, stats in metrics.items():
                if stats["mean"] < 0.5:  # Assuming metrics are 0-1 scale
                    suggestions.append({
                        "type": "performance",
                        "message": f"Low {metric} score ({stats['mean']:.2f}). Consider revising template."
                    })
        
        # Check trends
        if "trends" in analysis:
            for metric, trend in analysis["trends"].items():
                if trend["direction"] == "negative" and abs(trend["change"]) > 0.1:
                    suggestions.append({
                        "type": "trend",
                        "message": f"Declining {metric} performance. Consider reverting to earlier version."
                    })
        
        return {
            "template_name": template_name,
            "suggestions": suggestions,
            "analysis": analysis
        }
    
    def _load_template_results(self, template_name: str):
        """Load optimization results for a template.
        
        Args:
            template_name: Name of the template
        """
        results = []
        
        # Scan result files
        for filename in os.listdir(self.results_dir):
            if not filename.endswith('.json'):
                continue
                
            filepath = os.path.join(self.results_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    
                    if result.get("template_name") == template_name:
                        result_id = filename.replace('.json', '')
                        results.append({
                            "id": result_id,
                            "metrics": result.get("metrics", {}),
                            "timestamp": result.get("timestamp")
                        })
            except Exception as e:
                logger.warning(f"Error loading result file {filepath}: {e}")
        
        # Store results
        self.optimization_results[template_name] = results
    
    def get_best_template_version(self, template_name: str, metric: str = None) -> Optional[str]:
        """Get the best performing version of a template.
        
        Args:
            template_name: Base name of the template
            metric: Metric to optimize for (highest value wins)
            
        Returns:
            Best version name or None if no versions found
        """
        # Find all versions of the template
        template_versions = []
        pattern = re.compile(f"^{re.escape(template_name)}(\.v\\d+(\\.\\d+)?)?$")
        
        for name in self.library.templates.keys():
            if pattern.match(name):
                template_versions.append(name)
        
        if not template_versions:
            return None
            
        # If only one version, return it
        if len(template_versions) == 1:
            return template_versions[0]
        
        # Compare versions based on performance
        best_version = None
        best_score = -float('inf')
        
        for version in template_versions:
            analysis = self.analyze_template_performance(version)
            
            # Skip if no metrics
            if "metrics_stats" not in analysis or not analysis["metrics_stats"]:
                continue
                
            # Choose metric
            if metric is None:
                # Use first available metric
                available_metrics = list(analysis["metrics_stats"].keys())
                if not available_metrics:
                    continue
                metric_to_use = available_metrics[0]
            else:
                metric_to_use = metric
                if metric_to_use not in analysis["metrics_stats"]:
                    continue
            
            # Get score
            score = analysis["metrics_stats"][metric_to_use]["mean"]
            
            # Update best if better
            if score > best_score:
                best_score = score
                best_version = version
        
        return best_version

class DefaultPromptTemplates:
    """Default prompt templates for common use cases."""
    
    @staticmethod
    def get_default_templates() -> List[PromptTemplate]:
        """Get a list of default prompt templates.
        
        Returns:
            List of PromptTemplate instances
        """
        templates = []
        
        # RAG Query Template
        rag_query = PromptTemplate(
            name="rag_query",
            description="Template for RAG queries with context",
            template="""You are a helpful assistant that provides accurate information based on the given context.

Context:
{context}

User Question: {query}

Instructions:
1. Answer the question based only on the provided context.
2. If the context doesn't contain the information needed, respond with "I don't have enough information to answer this question."
3. Keep your answer concise and to the point.
4. If appropriate, cite specific parts of the context that support your answer.

Answer:""",
            variables=["context", "query"]
        )
        templates.append(rag_query)
        
        # Document Summarization Template
        summarization = PromptTemplate(
            name="document_summary",
            description="Template for document summarization",
            template="""Summarize the following document:

Document: {document}

Instructions:
1. Provide a comprehensive summary that captures the main points and key details.
2. The summary should be about {length} in length.
3. Focus on the most important information and maintain the original meaning.
4. Structure the summary in a logical and coherent manner.

Summary:""",
            variables=["document", "length"]
        )
        templates.append(summarization)
        
        # Entity Extraction Template
        entity_extraction = PromptTemplate(
            name="entity_extraction",
            description="Template for extracting entities from text",
            template="""Extract the entities from the following text:

Text: {text}

Entity types to extract:
{entity_types}

Instructions:
1. Identify all instances of the requested entity types in the text.
2. Format your response as a JSON list of objects with the following structure:
   [
     {"text": "entity text", "type": "entity type", "start": start_index, "end": end_index}
   ]
3. Make sure to correctly identify entity boundaries.
4. Only include entities explicitly mentioned in the text.

Extracted Entities:""",
            variables=["text", "entity_types"]
        )
        templates.append(entity_extraction)
        
        # Document Categorization Template
        categorization = PromptTemplate(
            name="document_categorization",
            description="Template for categorizing documents",
            template="""Categorize the following document:

Document: {document}

Available Categories:
{categories}

Instructions:
1. Analyze the document and assign the most appropriate category from the provided list.
2. If multiple categories apply, list them in order of relevance.
3. Provide a brief explanation for your categorization.
4. If none of the categories are appropriate, suggest a new category.

Format your response as:
Category: [Primary category]
Additional Categories: [Other relevant categories, if any]
Explanation: [Brief explanation for the categorization]

Response:""",
            variables=["document", "categories"]
        )
        templates.append(categorization)
        
        # Query Expansion Template
        query_expansion = PromptTemplate(
            name="query_expansion",
            description="Template for expanding search queries",
            template="""Expand the following search query:

Original Query: {query}

Instructions:
1. Generate {num_expansions} alternative variations of the original query.
2. Include synonyms, related concepts, and different phrasings.
3. Ensure the expanded queries maintain the original intent.
4. List each expanded query on a new line.
5. Do not include explanations, just list the expanded queries.

Expanded Queries:""",
            variables=["query", "num_expansions"]
        )
        templates.append(query_expansion)
        
        # Knowledge Graph Relationship Template
        relationship_extraction = PromptTemplate(
            name="relationship_extraction",
            description="Template for extracting relationships for knowledge graphs",
            template="""Extract relationships between entities in the following text:

Text: {text}

Entities Found:
{entities}

Instructions:
1. Identify relationships between the provided entities in the text.
2. Format your response as a JSON list of objects with the following structure:
   [
     {"subject": "entity1", "predicate": "relationship type", "object": "entity2", "confidence": 0.0-1.0}
   ]
3. Only include relationships that are explicitly stated in the text.
4. Assign a confidence score (between 0.0 and 1.0) to each relationship.

Extracted Relationships:""",
            variables=["text", "entities"]
        )
        templates.append(relationship_extraction)
        
        # Fact Verification Template
        fact_verification = PromptTemplate(
            name="fact_verification",
            description="Template for verifying facts against context",
            template="""Verify the following statement against the provided context:

Statement: {statement}

Context:
{context}

Instructions:
1. Determine whether the statement is supported, contradicted, or unaddressed by the context.
2. If supported, cite the specific parts of the context that support it.
3. If contradicted, cite the specific parts of the context that contradict it.
4. If the context doesn't address the statement, indicate that it's unaddressed.

Verification Result:
Judgment: [Supported/Contradicted/Unaddressed]
Evidence: [Relevant portions of the context]
Explanation: [Brief explanation of your judgment]""",
            variables=["statement", "context"]
        )
        templates.append(fact_verification)
        
        return templates

def initialize_prompt_management(config: Dict[str, Any] = None) -> PromptLibrary:
    """Initialize the prompt management system.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Initialized PromptLibrary instance
    """
    # Get config settings
    prompt_dir = "data/prompts"
    if config and "llm" in config and "prompt_management" in config["llm"]:
        prompt_dir = config["llm"]["prompt_management"].get("library_path", prompt_dir)
    
    # Initialize library
    library = PromptLibrary(library_path=prompt_dir)
    
    # Check if we need to add default templates
    if not library.templates:
        # Add default templates
        templates = DefaultPromptTemplates.get_default_templates()
        for template in templates:
            library.add_template(template)
    
    return library

# Global instances
prompt_library = None
prompt_optimizer = None

def get_prompt_library() -> PromptLibrary:
    """Get the global prompt library instance.
    
    Returns:
        PromptLibrary instance
    """
    global prompt_library
    if prompt_library is None:
        prompt_library = initialize_prompt_management()
    return prompt_library

def get_prompt_optimizer() -> PromptOptimizer:
    """Get the global prompt optimizer instance.
    
    Returns:
        PromptOptimizer instance
    """
    global prompt_optimizer
    if prompt_optimizer is None:
        prompt_optimizer = PromptOptimizer(get_prompt_library())
    return prompt_optimizer

def format_prompt(template_name: str, **kwargs) -> str:
    """Format a prompt using a template from the global library.
    
    Args:
        template_name: Name of the template to use
        **kwargs: Variable values for substitution
        
    Returns:
        Formatted prompt string
    """
    return get_prompt_library().format_prompt(template_name, **kwargs)
