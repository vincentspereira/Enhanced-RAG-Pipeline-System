"""
Prompt management system integration for the RAG pipeline.

This module integrates the prompt management system with the RAG pipeline,
providing a comprehensive way to manage, version, optimize, and analyze prompts.
"""
import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import yaml
from datetime import datetime

from .prompt_management import PromptTemplate, PromptLibrary, PromptOptimizer
from ..utils.analytics import AnalyticsTracker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PromptManagementSystem:
    """System for managing prompts in the RAG pipeline."""
    
    def __init__(
        self,
        library_path: str = "data/prompts",
        analytics_path: str = "data/prompt_analytics",
        optimization_enabled: bool = True
    ):
        """Initialize the prompt management system.
        
        Args:
            library_path: Path to the prompt library
            analytics_path: Path to store prompt analytics
            optimization_enabled: Whether to enable prompt optimization
        """
        self.library = PromptLibrary(library_path)
        self.optimizer = PromptOptimizer() if optimization_enabled else None
        self.analytics = PromptAnalytics(analytics_path)
        
        # Initialize default templates if needed
        self._initialize_defaults()
    
    def _initialize_defaults(self):
        """Initialize default templates if they don't exist."""
        from .prompt_management import DefaultPromptTemplates
        
        for name, template_data in DefaultPromptTemplates.items():
            if not self.library.get_template(name):
                template = PromptTemplate(
                    template=template_data["template"],
                    name=name,
                    description=template_data.get("description", ""),
                    variables=template_data.get("variables")
                )
                self.library.add_template(template)
                logger.info(f"Added default template: {name}")
    
    def get_prompt(self, template_name: str, **kwargs) -> str:
        """Get a formatted prompt using a template.
        
        Args:
            template_name: Name of the template to use
            **kwargs: Variables for prompt formatting
            
        Returns:
            Formatted prompt string
            
        Raises:
            ValueError: If template not found or formatting error
        """
        template = self.library.get_template(template_name)
        if not template:
            raise ValueError(f"Prompt template '{template_name}' not found")
        
        # Format the prompt
        prompt = template.format(**kwargs)
        
        # Track usage
        self.analytics.track_usage(template_name, kwargs)
        
        return prompt
    
    def optimize_prompt(self, template_name: str, context: Dict[str, Any] = None) -> str:
        """Optimize a prompt template for specific context.
        
        Args:
            template_name: Name of the template to optimize
            context: Context information for optimization
            
        Returns:
            Optimized prompt string
            
        Raises:
            ValueError: If template not found or optimization error
        """
        if not self.optimizer:
            raise ValueError("Prompt optimization is disabled")
        
        template = self.library.get_template(template_name)
        if not template:
            raise ValueError(f"Prompt template '{template_name}' not found")
        
        # Optimize the template
        optimized_template = self.optimizer.optimize(template, context)
        
        # Track optimization
        self.analytics.track_optimization(template_name, context)
        
        return optimized_template
    
    def create_template(self, template: str, name: str, description: str = None, 
                       variables: List[str] = None) -> PromptTemplate:
        """Create a new prompt template.
        
        Args:
            template: Template string
            name: Template name
            description: Optional description
            variables: Optional variable list
            
        Returns:
            New PromptTemplate instance
            
        Raises:
            ValueError: If template already exists or creation error
        """
        if self.library.get_template(name):
            raise ValueError(f"Template '{name}' already exists")
        
        # Create template
        prompt_template = PromptTemplate(
            template=template,
            name=name,
            description=description,
            variables=variables
        )
        
        # Add to library
        success = self.library.add_template(prompt_template)
        if not success:
            raise ValueError(f"Failed to create template '{name}'")
        
        return prompt_template
    
    def update_template(self, name: str, template: str = None, description: str = None, 
                       variables: List[str] = None) -> PromptTemplate:
        """Update an existing prompt template.
        
        Args:
            name: Template name
            template: New template string (optional)
            description: New description (optional)
            variables: New variables list (optional)
            
        Returns:
            Updated PromptTemplate instance
            
        Raises:
            ValueError: If template not found or update error
        """
        existing_template = self.library.get_template(name)
        if not existing_template:
            raise ValueError(f"Template '{name}' not found")
        
        # Update fields
        if template is not None:
            existing_template.template = template
        if description is not None:
            existing_template.description = description
        if variables is not None:
            existing_template.variables = variables
        
        # Validate after updates
        existing_template._validate_template()
        
        # Update in library
        success = self.library.update_template(name, existing_template)
        if not success:
            raise ValueError(f"Failed to update template '{name}'")
        
        return existing_template
    
    def get_performance_metrics(self, template_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics for prompts.
        
        Args:
            template_name: Optional name to filter metrics for a specific template
            
        Returns:
            Dictionary of performance metrics
        """
        return self.analytics.get_performance_metrics(template_name)
    
    def get_usage_stats(self, template_name: Optional[str] = None) -> Dict[str, Any]:
        """Get usage statistics for prompts.
        
        Args:
            template_name: Optional name to filter stats for a specific template
            
        Returns:
            Dictionary of usage statistics
        """
        return self.analytics.get_usage_stats(template_name)


class PromptAnalytics:
    """Analytics for prompt usage and performance."""
    
    def __init__(self, analytics_path: str = "data/prompt_analytics"):
        """Initialize prompt analytics.
        
        Args:
            analytics_path: Path to store analytics data
        """
        self.analytics_path = analytics_path
        os.makedirs(analytics_path, exist_ok=True)
        
        # Initialize analytics files if they don't exist
        self.usage_file = os.path.join(analytics_path, "usage.json")
        self.performance_file = os.path.join(analytics_path, "performance.json")
        self.optimization_file = os.path.join(analytics_path, "optimization.json")
        
        # Load existing data
        self.usage_data = self._load_json(self.usage_file, {"templates": {}, "total_usage": 0})
        self.performance_data = self._load_json(self.performance_file, {"templates": {}})
        self.optimization_data = self._load_json(self.optimization_file, {"templates": {}})
    
    def _load_json(self, filepath: str, default: Dict) -> Dict:
        """Load JSON data or return default if file doesn't exist.
        
        Args:
            filepath: Path to JSON file
            default: Default data if file doesn't exist
            
        Returns:
            Loaded data or default
        """
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
        
        return default
    
    def _save_json(self, filepath: str, data: Dict) -> bool:
        """Save data to JSON file.
        
        Args:
            filepath: Path to JSON file
            data: Data to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving to {filepath}: {e}")
            return False
    
    def track_usage(self, template_name: str, variables: Dict[str, Any]):
        """Track usage of a prompt template.
        
        Args:
            template_name: Name of the template
            variables: Variables used for formatting
        """
        timestamp = datetime.now().isoformat()
        
        # Initialize template data if not exists
        if template_name not in self.usage_data["templates"]:
            self.usage_data["templates"][template_name] = {
                "count": 0,
                "recent_usages": []
            }
        
        # Update usage count
        self.usage_data["templates"][template_name]["count"] += 1
        self.usage_data["total_usage"] += 1
        
        # Add to recent usages (keep last 20)
        recent = self.usage_data["templates"][template_name]["recent_usages"]
        recent.append({
            "timestamp": timestamp,
            "variables": {k: str(v)[:100] for k, v in variables.items()}  # Truncate long values
        })
        
        if len(recent) > 20:
            recent.pop(0)
        
        # Save updated data
        self._save_json(self.usage_file, self.usage_data)
    
    def track_optimization(self, template_name: str, context: Dict[str, Any]):
        """Track optimization of a prompt template.
        
        Args:
            template_name: Name of the template
            context: Context information used for optimization
        """
        timestamp = datetime.now().isoformat()
        
        # Initialize template data if not exists
        if template_name not in self.optimization_data["templates"]:
            self.optimization_data["templates"][template_name] = {
                "count": 0,
                "recent_optimizations": []
            }
        
        # Update optimization count
        self.optimization_data["templates"][template_name]["count"] += 1
        
        # Add to recent optimizations (keep last 10)
        recent = self.optimization_data["templates"][template_name]["recent_optimizations"]
        recent.append({
            "timestamp": timestamp,
            "context": {k: str(v)[:100] for k, v in context.items()}  # Truncate long values
        })
        
        if len(recent) > 10:
            recent.pop(0)
        
        # Save updated data
        self._save_json(self.optimization_file, self.optimization_data)
    
    def track_performance(self, template_name: str, metrics: Dict[str, Any]):
        """Track performance metrics for a prompt.
        
        Args:
            template_name: Name of the template
            metrics: Performance metrics
        """
        timestamp = datetime.now().isoformat()
        
        # Initialize template data if not exists
        if template_name not in self.performance_data["templates"]:
            self.performance_data["templates"][template_name] = {
                "metrics_count": 0,
                "cumulative_metrics": {},
                "recent_metrics": []
            }
        
        template_data = self.performance_data["templates"][template_name]
        
        # Update metrics count
        template_data["metrics_count"] += 1
        
        # Update cumulative metrics
        for metric_name, value in metrics.items():
            if metric_name not in template_data["cumulative_metrics"]:
                template_data["cumulative_metrics"][metric_name] = 0
            
            # Only update if numeric
            if isinstance(value, (int, float)):
                template_data["cumulative_metrics"][metric_name] += value
        
        # Add to recent metrics (keep last 20)
        recent = template_data["recent_metrics"]
        recent.append({
            "timestamp": timestamp,
            "metrics": metrics
        })
        
        if len(recent) > 20:
            recent.pop(0)
        
        # Save updated data
        self._save_json(self.performance_file, self.performance_data)
    
    def get_performance_metrics(self, template_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics for prompts.
        
        Args:
            template_name: Optional name to filter metrics for a specific template
            
        Returns:
            Dictionary of performance metrics
        """
        if template_name:
            # Return metrics for specific template
            if template_name in self.performance_data["templates"]:
                template_data = self.performance_data["templates"][template_name]
                # Calculate averages
                avg_metrics = {}
                for metric_name, cumulative in template_data["cumulative_metrics"].items():
                    if template_data["metrics_count"] > 0:
                        avg_metrics[f"avg_{metric_name}"] = cumulative / template_data["metrics_count"]
                
                return {
                    "template": template_name,
                    "metrics_count": template_data["metrics_count"],
                    "average_metrics": avg_metrics,
                    "recent_metrics": template_data["recent_metrics"]
                }
            else:
                return {"error": f"Template '{template_name}' not found in performance data"}
        else:
            # Return summary metrics for all templates
            summary = {
                "templates": [],
                "total_metrics_count": sum(t["metrics_count"] for t in self.performance_data["templates"].values())
            }
            
            for name, data in self.performance_data["templates"].items():
                # Calculate averages
                avg_metrics = {}
                for metric_name, cumulative in data["cumulative_metrics"].items():
                    if data["metrics_count"] > 0:
                        avg_metrics[f"avg_{metric_name}"] = cumulative / data["metrics_count"]
                
                summary["templates"].append({
                    "name": name,
                    "metrics_count": data["metrics_count"],
                    "average_metrics": avg_metrics
                })
            
            return summary
    
    def get_usage_stats(self, template_name: Optional[str] = None) -> Dict[str, Any]:
        """Get usage statistics for prompts.
        
        Args:
            template_name: Optional name to filter stats for a specific template
            
        Returns:
            Dictionary of usage statistics
        """
        if template_name:
            # Return stats for specific template
            if template_name in self.usage_data["templates"]:
                template_data = self.usage_data["templates"][template_name]
                return {
                    "template": template_name,
                    "usage_count": template_data["count"],
                    "percentage_of_total": template_data["count"] / max(1, self.usage_data["total_usage"]) * 100,
                    "recent_usages": template_data["recent_usages"]
                }
            else:
                return {"error": f"Template '{template_name}' not found in usage data"}
        else:
            # Return summary stats for all templates
            summary = {
                "templates": [],
                "total_usage": self.usage_data["total_usage"]
            }
            
            for name, data in self.usage_data["templates"].items():
                summary["templates"].append({
                    "name": name,
                    "usage_count": data["count"],
                    "percentage_of_total": data["count"] / max(1, self.usage_data["total_usage"]) * 100
                })
            
            # Sort by usage count (descending)
            summary["templates"].sort(key=lambda x: x["usage_count"], reverse=True)
            
            return summary


# Helper function to get prompt management system instance
_prompt_management_system = None

def get_prompt_management_system() -> PromptManagementSystem:
    """Get the prompt management system instance."""
    global _prompt_management_system
    if _prompt_management_system is None:
        _prompt_management_system = PromptManagementSystem()
    return _prompt_management_system
