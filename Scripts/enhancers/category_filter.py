"""
Category-based search filter for RAG pipeline.

This module implements filtering of search results based on document categories.
"""
from typing import List, Dict, Any, Optional, Union, Set
from dataclasses import dataclass
import logging
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class CategoryFilterConfig:
    """Configuration for category-based search filtering."""
    enable_hierarchical_filtering: bool = True
    max_categories: int = 100
    min_confidence: float = 0.5
    category_weight: float = 0.3
    boost_exact_category_match: bool = True
    exact_match_boost: float = 2.0
    persist_category_stats: bool = True
    stats_file: str = "data/category_stats.json"
    enable_category_synonyms: bool = True
    category_synonyms: Dict[str, List[str]] = None


class CategoryFilter:
    """Filter search results based on document categories."""
    
    def __init__(self, config: CategoryFilterConfig = None):
        """Initialize category filter.
        
        Args:
            config: Configuration for category filtering
        """
        self.config = config or CategoryFilterConfig()
        self.category_stats = defaultdict(int)
        self.category_hierarchy = {}
        self.category_synonyms = {}
        
        if self.config.enable_category_synonyms and self.config.category_synonyms:
            self.category_synonyms = self.config.category_synonyms
        
        if self.config.persist_category_stats:
            self._load_category_stats()
    
    def _load_category_stats(self):
        """Load category statistics from disk."""
        stats_path = Path(self.config.stats_file)
        if not stats_path.exists():
            return
        
        try:
            with open(stats_path, 'r') as f:
                stats_data = json.load(f)
            
            self.category_stats = defaultdict(int, stats_data.get('stats', {}))
            self.category_hierarchy = stats_data.get('hierarchy', {})
        except Exception as e:
            logger.warning(f"Failed to load category stats: {e}")
    
    def _save_category_stats(self):
        """Save category statistics to disk."""
        if not self.config.persist_category_stats:
            return
        
        stats_path = Path(self.config.stats_file)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            stats_data = {
                'stats': dict(self.category_stats),
                'hierarchy': self.category_hierarchy
            }
            
            with open(stats_path, 'w') as f:
                json.dump(stats_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save category stats: {e}")
    
    def update_category_stats(self, results: List[Dict[str, Any]]):
        """Update category statistics based on search results.
        
        Args:
            results: Search results with category information
        """
        for result in results:
            metadata = result.get('metadata', {})
            categories = metadata.get('categories', [])
            
            if isinstance(categories, str):
                categories = [categories]
            
            for category in categories:
                if isinstance(category, dict):
                    category_id = category.get('category_id')
                    if category_id:
                        self.category_stats[category_id] += 1
                elif isinstance(category, str):
                    self.category_stats[category] += 1
        
        # Save updated stats
        self._save_category_stats()
    
    def filter_by_categories(
        self,
        results: List[Dict[str, Any]],
        categories: Union[List[str], str]
    ) -> List[Dict[str, Any]]:
        """Filter search results by categories.
        
        Args:
            results: Search results to filter
            categories: Category or list of categories to filter by
            
        Returns:
            Filtered search results
        """
        if not categories:
            return results
        
        if isinstance(categories, str):
            categories = [categories]
        
        # Expand categories with synonyms
        expanded_categories = set(categories)
        if self.config.enable_category_synonyms:
            for category in categories:
                synonyms = self.category_synonyms.get(category, [])
                expanded_categories.update(synonyms)
        
        # Get parent categories if hierarchical filtering is enabled
        if self.config.enable_hierarchical_filtering:
            parent_categories = set()
            for category in expanded_categories:
                parent = self.category_hierarchy.get(category)
                if parent:
                    parent_categories.add(parent)
            expanded_categories.update(parent_categories)
        
        filtered_results = []
        
        for result in results:
            metadata = result.get('metadata', {})
            result_categories = metadata.get('categories', [])
            
            # Handle string or list of strings
            if isinstance(result_categories, str):
                result_categories = [result_categories]
            
            # Handle list of category objects
            extracted_categories = set()
            for category in result_categories:
                if isinstance(category, dict):
                    category_id = category.get('category_id')
                    if category_id:
                        extracted_categories.add(category_id)
                        # Add subcategories if they exist
                        subcategories = category.get('subcategories', [])
                        if isinstance(subcategories, list):
                            for subcat in subcategories:
                                if isinstance(subcat, dict):
                                    subcat_id = subcat.get('category_id')
                                    if subcat_id:
                                        extracted_categories.add(subcat_id)
                elif isinstance(category, str):
                    extracted_categories.add(category)
            
            # Check if there's an intersection between result categories and filter categories
            if expanded_categories & extracted_categories:
                # Apply category boost if exact match
                exact_matches = expanded_categories & extracted_categories
                if self.config.boost_exact_category_match and exact_matches:
                    boost_factor = self.config.exact_match_boost
                    original_score = result.get('score', 0)
                    result['score'] = original_score * boost_factor
                    result['category_match'] = True
                    result['matched_categories'] = list(exact_matches)
                
                filtered_results.append(result)
        
        return filtered_results
    
    def suggest_categories(self, prefix: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Suggest categories based on prefix and usage statistics.
        
        Args:
            prefix: Category prefix to filter by
            limit: Maximum number of suggestions
            
        Returns:
            List of category suggestions
        """
        suggestions = []
        
        # Sort categories by usage count
        sorted_categories = sorted(
            self.category_stats.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Filter by prefix if provided
        if prefix:
            prefix = prefix.lower()
            filtered_categories = [
                (cat, count) for cat, count in sorted_categories
                if cat.lower().startswith(prefix)
            ]
        else:
            filtered_categories = sorted_categories
        
        # Build suggestions with hierarchy info
        for category, count in filtered_categories[:limit]:
            suggestion = {
                'id': category,
                'count': count,
                'parent': self.category_hierarchy.get(category)
            }
            
            # Add children info
            children = [
                child for child, parent in self.category_hierarchy.items()
                if parent == category
            ]
            if children:
                suggestion['children'] = children
            
            # Add synonyms
            synonyms = self.category_synonyms.get(category, [])
            if synonyms:
                suggestion['synonyms'] = synonyms
            
            suggestions.append(suggestion)
        
        return suggestions
    
    def update_category_hierarchy(self, hierarchy: Dict[str, str]):
        """Update category hierarchy.
        
        Args:
            hierarchy: Dictionary mapping category ID to parent category ID
        """
        self.category_hierarchy.update(hierarchy)
        self._save_category_stats()
    
    def update_category_synonyms(self, synonyms: Dict[str, List[str]]):
        """Update category synonyms.
        
        Args:
            synonyms: Dictionary mapping category ID to list of synonyms
        """
        self.category_synonyms.update(synonyms)
        self._save_category_stats()
    
    def get_popular_categories(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most popular categories based on usage.
        
        Args:
            limit: Maximum number of categories to return
            
        Returns:
            List of popular categories with usage counts
        """
        sorted_categories = sorted(
            self.category_stats.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return [
            {'id': cat, 'count': count}
            for cat, count in sorted_categories
        ]
    
    def reset_category_stats(self):
        """Reset category statistics."""
        self.category_stats = defaultdict(int)
        self._save_category_stats()
