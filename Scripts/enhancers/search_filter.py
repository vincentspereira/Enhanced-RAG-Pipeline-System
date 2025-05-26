from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import re
from datetime import datetime
import logging
import operator
import numpy as np
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class FilterConfig:
    max_filters: int = 10
    enable_range_filters: bool = True
    enable_regex_filters: bool = True
    case_sensitive: bool = False
    enable_nested_filters: bool = True
    enable_date_filters: bool = True
    date_formats: List[str] = None
    enable_numerical_comparison: bool = True
    batch_size: int = 1000

class FilterOperator:
    EQ = "eq"  # equals
    NEQ = "neq"  # not equals
    GT = "gt"  # greater than
    LT = "lt"  # less than
    GTE = "gte"  # greater than or equals
    LTE = "lte"  # less than or equals
    IN = "in"  # in list
    NIN = "nin"  # not in list
    CONTAINS = "contains"  # contains string
    NOT_CONTAINS = "not_contains"  # does not contain string
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX = "regex"  # matches regex pattern
    EXISTS = "exists"  # field exists
    NOT_EXISTS = "not_exists"  # field does not exist
    BETWEEN = "between"  # between two values

class SearchResultFilter:
    def __init__(self, config: FilterConfig = None):
        self.config = config or FilterConfig()
        if self.config.date_formats is None:
            self.config.date_formats = [
                "%Y-%m-%d",
                "%Y/%m/%d",
                "%d-%m-%Y",
                "%d/%m/%Y",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%fZ"
            ]

    def apply_filters(
        self,
        results: List[Dict[str, Any]],
        filters: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply multiple filters to search results"""
        if not filters or not results:
            return results

        # Validate filters
        if len(filters) > self.config.max_filters:
            logger.warning(f"Number of filters exceeds maximum ({self.config.max_filters})")
            filters = filters[:self.config.max_filters]

        # Process results in batches
        filtered_results = []
        for i in range(0, len(results), self.config.batch_size):
            batch = results[i:i + self.config.batch_size]
            
            # Process each result in the batch
            with ThreadPoolExecutor() as executor:
                filtered_batch = list(filter(
                    lambda x: self._matches_all_filters(x, filters),
                    batch
                ))
            filtered_results.extend(filtered_batch)

        return filtered_results

    def _matches_all_filters(
        self,
        result: Dict[str, Any],
        filters: List[Dict[str, Any]]
    ) -> bool:
        """Check if a result matches all filters"""
        return all(self._apply_filter(result, filter_dict)
                  for filter_dict in filters)

    def _apply_filter(
        self,
        result: Dict[str, Any],
        filter_dict: Dict[str, Any]
    ) -> bool:
        """Apply a single filter to a result"""
        field = filter_dict.get("field")
        operator = filter_dict.get("operator")
        value = filter_dict.get("value")
        
        if not field or not operator:
            return True

        try:
            field_value = self._get_field_value(result, field)
            
            # Handle non-existent fields
            if field_value is None:
                return operator in [FilterOperator.NOT_EXISTS]
                
            return self._compare_values(field_value, operator, value)
        except Exception as e:
            logger.error(f"Filter application failed: {e}")
            return True

    def _get_field_value(
        self,
        obj: Dict[str, Any],
        field: str
    ) -> Any:
        """Get value from potentially nested field"""
        if not self.config.enable_nested_filters:
            return obj.get(field)

        try:
            current = obj
            for part in field.split('.'):
                if isinstance(current, dict):
                    current = current.get(part)
                else:
                    return None
            return current
        except:
            return None

    def _compare_values(
        self,
        field_value: Any,
        operator: str,
        filter_value: Any
    ) -> bool:
        """Compare values based on operator"""
        # Handle different value types
        if self.config.enable_date_filters and self._is_date(field_value):
            return self._compare_dates(field_value, operator, filter_value)
            
        if self.config.enable_numerical_comparison and self._is_numeric(field_value):
            return self._compare_numbers(field_value, operator, filter_value)

        # String comparisons
        if isinstance(field_value, str):
            if not self.config.case_sensitive:
                field_value = field_value.lower()
                if isinstance(filter_value, str):
                    filter_value = filter_value.lower()
                elif isinstance(filter_value, list):
                    filter_value = [v.lower() for v in filter_value if isinstance(v, str)]

        # Apply comparison based on operator
        if operator == FilterOperator.EQ:
            return field_value == filter_value
        elif operator == FilterOperator.NEQ:
            return field_value != filter_value
        elif operator == FilterOperator.GT:
            return field_value > filter_value
        elif operator == FilterOperator.LT:
            return field_value < filter_value
        elif operator == FilterOperator.GTE:
            return field_value >= filter_value
        elif operator == FilterOperator.LTE:
            return field_value <= filter_value
        elif operator == FilterOperator.IN:
            return field_value in filter_value
        elif operator == FilterOperator.NIN:
            return field_value not in filter_value
        elif operator == FilterOperator.CONTAINS:
            return str(filter_value) in str(field_value)
        elif operator == FilterOperator.NOT_CONTAINS:
            return str(filter_value) not in str(field_value)
        elif operator == FilterOperator.STARTS_WITH:
            return str(field_value).startswith(str(filter_value))
        elif operator == FilterOperator.ENDS_WITH:
            return str(field_value).endswith(str(filter_value))
        elif operator == FilterOperator.REGEX and self.config.enable_regex_filters:
            try:
                return bool(re.match(filter_value, str(field_value)))
            except:
                return False
        elif operator == FilterOperator.EXISTS:
            return True
        elif operator == FilterOperator.NOT_EXISTS:
            return False
        elif operator == FilterOperator.BETWEEN:
            if isinstance(filter_value, (list, tuple)) and len(filter_value) == 2:
                return filter_value[0] <= field_value <= filter_value[1]
            return False
        
        return True

    def _is_date(self, value: Any) -> bool:
        """Check if value is or can be converted to a date"""
        if isinstance(value, datetime):
            return True
            
        if isinstance(value, str):
            for date_format in self.config.date_formats:
                try:
                    datetime.strptime(value, date_format)
                    return True
                except ValueError:
                    continue
        return False

    def _compare_dates(
        self,
        field_value: Union[str, datetime],
        operator: str,
        filter_value: Union[str, datetime]
    ) -> bool:
        """Compare date values"""
        # Convert string dates to datetime objects
        if isinstance(field_value, str):
            for date_format in self.config.date_formats:
                try:
                    field_value = datetime.strptime(field_value, date_format)
                    break
                except ValueError:
                    continue
                    
        if isinstance(filter_value, str):
            for date_format in self.config.date_formats:
                try:
                    filter_value = datetime.strptime(filter_value, date_format)
                    break
                except ValueError:
                    continue

        if not isinstance(field_value, datetime) or not isinstance(filter_value, datetime):
            return False

        # Compare dates
        ops = {
            FilterOperator.EQ: operator.eq,
            FilterOperator.NEQ: operator.ne,
            FilterOperator.GT: operator.gt,
            FilterOperator.LT: operator.lt,
            FilterOperator.GTE: operator.ge,
            FilterOperator.LTE: operator.le,
        }
        
        return ops.get(operator, lambda x, y: True)(field_value, filter_value)

    def _is_numeric(self, value: Any) -> bool:
        """Check if value is numeric"""
        return isinstance(value, (int, float, np.number))

    def _compare_numbers(
        self,
        field_value: Union[int, float],
        operator: str,
        filter_value: Union[int, float]
    ) -> bool:
        """Compare numeric values"""
        try:
            field_value = float(field_value)
            filter_value = float(filter_value)
        except (TypeError, ValueError):
            return False

        ops = {
            FilterOperator.EQ: operator.eq,
            FilterOperator.NEQ: operator.ne,
            FilterOperator.GT: operator.gt,
            FilterOperator.LT: operator.lt,
            FilterOperator.GTE: operator.ge,
            FilterOperator.LTE: operator.le,
        }
        
        return ops.get(operator, lambda x, y: True)(field_value, filter_value)

    def validate_filters(
        self,
        filters: List[Dict[str, Any]]
    ) -> Tuple[bool, List[str]]:
        """Validate filter configuration"""
        errors = []
        
        if len(filters) > self.config.max_filters:
            errors.append(f"Too many filters. Maximum allowed: {self.config.max_filters}")
            
        for filter_dict in filters:
            if "field" not in filter_dict:
                errors.append("Missing 'field' in filter")
            if "operator" not in filter_dict:
                errors.append("Missing 'operator' in filter")
            if "value" not in filter_dict and filter_dict.get("operator") not in [
                FilterOperator.EXISTS,
                FilterOperator.NOT_EXISTS
            ]:
                errors.append("Missing 'value' in filter")
            if filter_dict.get("operator") not in vars(FilterOperator).values():
                errors.append(f"Invalid operator: {filter_dict.get('operator')}")
                
        return len(errors) == 0, errors

    def suggest_filters(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze results and suggest possible filters"""
        if not results:
            return {}

        suggestions = {}
        sample_size = min(len(results), 100)
        sample = results[:sample_size]

        # Analyze fields and their types
        field_types = {}
        field_values = defaultdict(set)

        for result in sample:
            self._analyze_fields(result, "", field_types, field_values)

        # Generate suggestions for each field
        for field, type_info in field_types.items():
            if type_info["type"] == "date":
                suggestions[field] = {
                    "type": "date",
                    "operators": [
                        FilterOperator.EQ,
                        FilterOperator.GT,
                        FilterOperator.LT,
                        FilterOperator.BETWEEN
                    ],
                    "range": {
                        "min": min(field_values[field]),
                        "max": max(field_values[field])
                    } if field_values[field] else None
                }
            elif type_info["type"] == "number":
                suggestions[field] = {
                    "type": "number",
                    "operators": [
                        FilterOperator.EQ,
                        FilterOperator.GT,
                        FilterOperator.LT,
                        FilterOperator.BETWEEN
                    ],
                    "range": {
                        "min": min(field_values[field]),
                        "max": max(field_values[field])
                    } if field_values[field] else None
                }
            elif type_info["type"] == "string":
                unique_values = list(field_values[field])
                suggestions[field] = {
                    "type": "string",
                    "operators": [
                        FilterOperator.EQ,
                        FilterOperator.CONTAINS,
                        FilterOperator.STARTS_WITH,
                        FilterOperator.REGEX
                    ],
                    "unique_values": unique_values[:10] if len(unique_values) <= 10 else None,
                    "has_many_values": len(unique_values) > 10
                }
            elif type_info["type"] == "boolean":
                suggestions[field] = {
                    "type": "boolean",
                    "operators": [FilterOperator.EQ],
                    "values": [True, False]
                }
            elif type_info["type"] == "array":
                suggestions[field] = {
                    "type": "array",
                    "operators": [
                        FilterOperator.CONTAINS,
                        FilterOperator.IN,
                        FilterOperator.NIN
                    ]
                }

        return suggestions

    def _analyze_fields(
        self,
        obj: Any,
        prefix: str,
        field_types: Dict[str, Dict[str, str]],
        field_values: Dict[str, set]
    ):
        """Recursively analyze fields in a result object"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                self._analyze_fields(value, new_prefix, field_types, field_values)
        else:
            if self._is_date(obj):
                field_types[prefix] = {"type": "date"}
                if isinstance(obj, datetime):
                    field_values[prefix].add(obj)
            elif self._is_numeric(obj):
                field_types[prefix] = {"type": "number"}
                field_values[prefix].add(float(obj))
            elif isinstance(obj, bool):
                field_types[prefix] = {"type": "boolean"}
                field_values[prefix].add(obj)
            elif isinstance(obj, (list, tuple)):
                field_types[prefix] = {"type": "array"}
            elif isinstance(obj, str):
                field_types[prefix] = {"type": "string"}
                field_values[prefix].add(obj)
