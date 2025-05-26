from typing import List, Dict, Any, Optional, Union, Set
from dataclasses import dataclass
import logging
from collections import defaultdict
import re
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import json

logger = logging.getLogger(__name__)

@dataclass
class FacetConfig:
    enabled_facets: List[str] = None  # List of facet fields to compute
    max_facet_values: int = 100  # Maximum number of values per facet
    min_facet_count: int = 1  # Minimum count to include a facet value
    date_intervals: List[str] = None  # e.g., ["year", "month", "day"]
    numeric_intervals: Dict[str, List[float]] = None  # Custom numeric ranges
    enable_nested_facets: bool = False
    enable_statistics: bool = True
    case_sensitive: bool = False

class FacetedSearch:
    def __init__(self, config: FacetConfig = None):
        self.config = config or FacetConfig()
        if self.config.enabled_facets is None:
            self.config.enabled_facets = []
        if self.config.date_intervals is None:
            self.config.date_intervals = ["year", "month", "day"]
        if self.config.numeric_intervals is None:
            self.config.numeric_intervals = {}

    def extract_facets(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract facets from a list of documents"""
        facets = defaultdict(lambda: defaultdict(int))
        statistics = defaultdict(lambda: {
            "min": None,
            "max": None,
            "avg": None,
            "sum": 0,
            "count": 0
        })

        def process_document(doc: Dict[str, Any]):
            for field in self.config.enabled_facets:
                value = self._get_nested_value(doc, field)
                if value is None:
                    continue

                # Handle different value types
                if isinstance(value, (list, set, tuple)):
                    for v in value:
                        self._process_facet_value(facets[field], statistics[field], v)
                else:
                    self._process_facet_value(facets[field], statistics[field], value)

        # Process documents in parallel
        with ThreadPoolExecutor() as executor:
            executor.map(process_document, documents)

        # Finalize statistics
        if self.config.enable_statistics:
            for field in statistics:
                stats = statistics[field]
                if stats["count"] > 0:
                    stats["avg"] = stats["sum"] / stats["count"]

        # Format and filter results
        result = {}
        for field, values in facets.items():
            # Sort by count and apply minimum count filter
            filtered_values = {
                k: v for k, v in values.items()
                if v >= self.config.min_facet_count
            }
            
            # Limit number of values per facet
            sorted_values = dict(
                sorted(
                    filtered_values.items(),
                    key=lambda x: (-x[1], x[0])
                )[:self.config.max_facet_values]
            )
            
            result[field] = {
                "values": sorted_values,
                "statistics": statistics[field] if self.config.enable_statistics else None
            }

        return result

    def _process_facet_value(
        self,
        facet_counts: Dict[str, int],
        statistics: Dict[str, Any],
        value: Any
    ):
        """Process a single facet value"""
        if value is None:
            return

        # Convert value to string for facet counting
        if isinstance(value, bool):
            str_value = str(value).lower()
        elif isinstance(value, (int, float)):
            str_value = str(value)
            # Update numeric statistics
            if self.config.enable_statistics:
                statistics["sum"] += value
                statistics["count"] += 1
                if statistics["min"] is None or value < statistics["min"]:
                    statistics["min"] = value
                if statistics["max"] is None or value > statistics["max"]:
                    statistics["max"] = value
        elif isinstance(value, datetime):
            # Create date interval facets
            for interval in self.config.date_intervals:
                if interval == "year":
                    facet_counts[f"{value.year}"] += 1
                elif interval == "month":
                    facet_counts[f"{value.year}-{value.month:02d}"] += 1
                elif interval == "day":
                    facet_counts[f"{value.year}-{value.month:02d}-{value.day:02d}"] += 1
            return
        else:
            str_value = str(value)

        if not self.config.case_sensitive:
            str_value = str_value.lower()

        facet_counts[str_value] += 1

    def _get_nested_value(self, obj: Dict[str, Any], path: str) -> Any:
        """Get value from nested dictionary using dot notation"""
        if not self.config.enable_nested_facets:
            return obj.get(path)

        try:
            parts = path.split(".")
            current = obj
            for part in parts:
                if isinstance(current, dict):
                    current = current.get(part)
                else:
                    return None
            return current
        except:
            return None

    def apply_facet_filters(
        self,
        documents: List[Dict[str, Any]],
        filters: Dict[str, Union[List[str], Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Filter documents based on facet values"""
        if not filters:
            return documents

        def matches_filters(doc: Dict[str, Any]) -> bool:
            for field, filter_value in filters.items():
                value = self._get_nested_value(doc, field)
                if value is None:
                    return False

                if isinstance(filter_value, list):
                    # Handle multi-value filters
                    if isinstance(value, (list, set, tuple)):
                        if not any(v in filter_value for v in value):
                            return False
                    elif value not in filter_value:
                        return False
                elif isinstance(filter_value, dict):
                    # Handle range filters
                    try:
                        num_value = float(value)
                        if ("min" in filter_value and 
                            num_value < filter_value["min"]):
                            return False
                        if ("max" in filter_value and 
                            num_value > filter_value["max"]):
                            return False
                    except:
                        return False
                else:
                    # Handle single value filter
                    if isinstance(value, (list, set, tuple)):
                        if filter_value not in value:
                            return False
                    elif value != filter_value:
                        return False

            return True

        # Apply filters in parallel
        with ThreadPoolExecutor() as executor:
            filtered_docs = list(filter(matches_filters, documents))

        return filtered_docs

    def suggest_facet_values(
        self,
        field: str,
        prefix: str,
        facets: Dict[str, Any]
    ) -> List[str]:
        """Suggest facet values based on prefix"""
        if field not in facets:
            return []

        prefix = prefix.lower() if not self.config.case_sensitive else prefix
        suggestions = []
        
        for value in facets[field]["values"].keys():
            str_value = str(value)
            if not self.config.case_sensitive:
                str_value = str_value.lower()
            if str_value.startswith(prefix):
                suggestions.append(value)
                
        return suggestions[:self.config.max_facet_values]

    def analyze_facet_distribution(
        self,
        field: str,
        facets: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze distribution of values for a facet"""
        if field not in facets:
            return {}

        values = facets[field]["values"]
        if not values:
            return {}

        # Calculate distribution statistics
        total_count = sum(values.values())
        percentages = {k: (v / total_count) * 100 for k, v in values.items()}
        
        sorted_values = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "total_count": total_count,
            "unique_values": len(values),
            "top_values": dict(sorted_values[:10]),
            "distribution": percentages
        }

    def get_numeric_ranges(
        self,
        field: str,
        facets: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate numeric ranges for a numeric facet"""
        if field not in facets or not facets[field]["statistics"]:
            return []

        stats = facets[field]["statistics"]
        if stats["min"] is None or stats["max"] is None:
            return []

        # Use custom intervals if defined
        if field in self.config.numeric_intervals:
            ranges = []
            intervals = sorted(self.config.numeric_intervals[field])
            
            # Add range below first interval
            if stats["min"] < intervals[0]:
                ranges.append({
                    "min": stats["min"],
                    "max": intervals[0],
                    "label": f"Less than {intervals[0]}"
                })

            # Add ranges between intervals
            for i in range(len(intervals) - 1):
                ranges.append({
                    "min": intervals[i],
                    "max": intervals[i + 1],
                    "label": f"{intervals[i]} to {intervals[i + 1]}"
                })

            # Add range above last interval
            if stats["max"] > intervals[-1]:
                ranges.append({
                    "min": intervals[-1],
                    "max": stats["max"],
                    "label": f"More than {intervals[-1]}"
                })

            return ranges

        # Auto-generate ranges
        min_val = stats["min"]
        max_val = stats["max"]
        range_size = (max_val - min_val) / 10  # Create 10 ranges

        ranges = []
        for i in range(10):
            range_min = min_val + (i * range_size)
            range_max = min_val + ((i + 1) * range_size)
            ranges.append({
                "min": range_min,
                "max": range_max,
                "label": f"{range_min:.2f} to {range_max:.2f}"
            })

        return ranges
