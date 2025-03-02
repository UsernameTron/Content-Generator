"""
Path-based relationship encoding for healthcare metrics data.
Provides efficient encoding and decoding of nested data structures with relationship preservation.
"""

from typing import Dict, List, Any, Union


class PathEncoder:
    def encode(self, nested_dict: Dict[str, Any]) -> List[str]:
        """
        Encode nested dictionary with path-based relationships.
        
        Args:
            nested_dict: The nested dictionary to encode
            
        Returns:
            List of strings with path-based encoding
        """
        results = []
        
        # Special handling for metadata as recommended
        if 'metadata' in nested_dict:
            metadata = nested_dict['metadata']
            if isinstance(metadata, list):
                for item in metadata:
                    if isinstance(item, str) and ':' in item:
                        key, value = item.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        results.append(f"meta:metadata.{key} = {value}")
            elif isinstance(metadata, dict):
                for key, value in metadata.items():
                    results.append(f"meta:metadata.{key} = {value}")
        
        self._encode_recursive(nested_dict, "", results)
        return results
        
    def _encode_recursive(self, obj: Any, path: str, results: List[str], marker: str = "p") -> None:
        """
        Recursively encode nested dictionary with path-based encoding.
        
        Args:
            obj: The object to encode
            path: The current path
            results: The list to add results to
            marker: The marker to use for the current level
        """
        # Handle None values
        if obj is None:
            results.append(f"{marker}:{path} = None")
            return
            
        # Special handling for metadata - if this is a list of strings with format 'key: value'
        if path == 'metadata' and isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, str) and ':' in item:
                    key, value = item.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    results.append(f"meta:metadata.{key} = {value}")
                    
                    # Add relationships for specific metadata types
                    if key.lower() in ('type', 'category', 'domain', 'measure_type'):
                        results.append(f"rel:metadata.{key}->values.current = categorizes")
                    if key.lower() in ('date', 'timestamp', 'period'):
                        results.append(f"rel:metadata.{key}->values.current = time_context")
                else:
                    results.append(f"meta:metadata[{i}] = {item}")
            return
            
        # Check if this is a healthcare metrics structure
        is_metric_container = False
        if isinstance(obj, dict):
            # Check for common healthcare metric patterns
            has_values = 'values' in obj and isinstance(obj['values'], dict)
            has_metric_keys = any(k in obj for k in ('name', 'id', 'type', 'category'))
            
            if has_values and has_metric_keys:
                is_metric_container = True
                # Add explicit marker for metric container
                results.append(f"m:{path} = healthcare_metric")
                
                # Add special relationships for metrics
                if 'values' in obj and isinstance(obj['values'], dict):
                    values = obj['values']
                    # Add relationships between baseline, current, and target
                    if all(k in values for k in ('baseline', 'current', 'target')):
                        # Baseline to current relationship
                        baseline_path = f"{path}.values.baseline"
                        current_path = f"{path}.values.current"
                        results.append(f"rel:{baseline_path}->{current_path} = metric_improvement")
                        
                        # Current to target relationship
                        target_path = f"{path}.values.target"
                        results.append(f"rel:{current_path}->{target_path} = target_achievement_progress")
            
        # Process dictionary objects
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                
                # Determine appropriate marker based on key and context
                current_marker = marker
                
                # Healthcare metric-specific markers
                if key in ('reasoning', 'adaptability', 'integration', 'knowledge_integration'):
                    current_marker = 'm'  # metric
                elif key in ('relationship', 'correlation', 'impact'):
                    current_marker = 'rel'  # relationship
                elif key == 'values' and is_metric_container:
                    current_marker = 'm'  # metric values
                elif key == 'factors' and is_metric_container:
                    current_marker = 'f'  # factors
                elif key == 'components' and is_metric_container:
                    current_marker = 'c'  # components
                elif key in ('metadata', 'description', 'notes'):
                    current_marker = 'meta'  # metadata
                
                # Special handling for values in healthcare metrics
                if key == 'values' and is_metric_container and isinstance(value, dict):
                    # Process value fields with specific markers and relationships
                    for value_key, value_val in value.items():
                        value_path = f"{current_path}.{value_key}"
                        # Use metric marker for all value fields
                        results.append(f"m:{value_path} = {value_val}")
                        
                        # Add human-readable descriptions
                        if value_key == 'baseline':
                            results.append(f"meta:{value_path} = description: Initial metric value before improvements")
                        elif value_key == 'current':
                            results.append(f"meta:{value_path} = description: Current metric value after improvements")
                        elif value_key == 'target':
                            results.append(f"meta:{value_path} = description: Target metric value to achieve")
                    continue  # Skip normal processing since we've handled this field specially
                
                # Regular recursive processing
                if isinstance(value, dict):
                    self._encode_recursive(value, current_path, results, current_marker)
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        item_path = f"{current_path}[{i}]"
                        if isinstance(item, dict):
                            self._encode_recursive(item, item_path, results, current_marker)
                        else:
                            # For list items, preserve the current marker
                            results.append(f"{current_marker}:{item_path} = {item}")
                else:
                    # Enhanced value encoding for healthcare metrics
                    if is_metric_container and key in ('name', 'id', 'type', 'category'):
                        # These are metric metadata fields
                        results.append(f"meta:{current_path} = {value}")
                        # Add relationship to current value if it exists
                        if 'values' in obj and 'current' in obj['values']:
                            results.append(f"rel:{current_path}->{path}.values.current = describes")
                    else:
                        # Normal value encoding
                        results.append(f"{current_marker}:{current_path} = {value}")
        
        # Process list objects
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                item_path = f"{path}[{i}]"
                if isinstance(item, dict):
                    self._encode_recursive(item, item_path, results, marker)
                else:
                    results.append(f"{marker}:{item_path} = {item}")

    def encode_relationships(self, source_path: str, target_path: str, rel_type: str) -> str:
        """
        Create a relationship encoding between two paths.
        
        Args:
            source_path: The source path
            target_path: The target path
            rel_type: The relationship type
            
        Returns:
            A relationship encoding string
        """
        return f"rel:{source_path}->{target_path} = {rel_type}"

    def decode(self, encoded_paths: List[str]) -> Dict[str, Any]:
        """
        Decode path-based relationships back to nested structure.
        
        Args:
            encoded_paths: List of strings with path-based encoding
            
        Returns:
            Reconstructed nested dictionary
        """
        result = {}
        relationships = []
        
        for path in encoded_paths:
            if "=" not in path:
                continue
                
            parts = path.split("=", 1)
            path_part = parts[0].strip()
            value_part = parts[1].strip()
            
            # Strip marker prefix
            if ":" in path_part:
                marker, path_clean = path_part.split(":", 1)
                
                # Handle relationships separately
                if marker == "rel" and "->" in path_clean:
                    source, target = path_clean.split("->", 1)
                    relationships.append({
                        "source": source.strip(),
                        "target": target.strip(),
                        "type": value_part
                    })
                    continue
            else:
                path_clean = path_part
                
            self._set_nested_value(result, path_clean, value_part)
        
        # Add relationships to result if any exist
        if relationships:
            if "relationships" not in result:
                result["relationships"] = []
            result["relationships"].extend(relationships)
            
        return result
        
    def _set_nested_value(self, obj: Dict[str, Any], path: str, value: Any) -> None:
        """
        Set a value in a nested dictionary based on a path.
        
        Args:
            obj: The dictionary to set the value in
            path: The path to set the value at
            value: The value to set
        """
        parts = path.split(".")
        current = obj
        
        for i, part in enumerate(parts[:-1]):
            # Handle array indices in path
            if "[" in part:
                name, idx_part = part.split("[", 1)
                idx = int(idx_part.rstrip("]"))
                
                if name not in current:
                    current[name] = []
                while len(current[name]) <= idx:
                    current[name].append({})
                current = current[name][idx]
            else:
                if part not in current:
                    current[part] = {}
                current = current[part]
                
        last_part = parts[-1]
        if "[" in last_part:
            name, idx_part = last_part.split("[", 1)
            idx = int(idx_part.rstrip("]"))
            
            if name not in current:
                current[name] = []
            while len(current[name]) <= idx:
                current[name].append(None)
            current[name][idx] = value
        else:
            current[last_part] = value

    def encode_value(self, path: str, value: Any) -> str:
        """
        Encode a value with a specific path.
        
        Args:
            path: The path to encode the value with
            value: The value to encode
            
        Returns:
            An encoded string
        """
        # Determine appropriate marker based on path
        marker = 'p'  # Default is property
        if 'values.' in path:
            marker = 'm'  # Metric value
        elif 'relationship' in path or 'correlation' in path:
            marker = 'rel'  # Relationship
            
        # Convert value to string appropriately
        if isinstance(value, (int, float)):
            value_str = str(value)
        elif isinstance(value, dict) and 'value' in value:
            value_str = str(value['value'])
        elif isinstance(value, dict) and len(value) > 0:
            # Try to convert a simple dict to string representation
            value_str = str(value)
        else:
            value_str = str(value)
            
        return f"{marker}:{path} = {value_str}"
    
    def extract_relationship(self, encoded_path: str) -> Dict[str, str]:
        """
        Extract relationship information from an encoded path string.
        
        Args:
            encoded_path: The encoded path string
            
        Returns:
            Dictionary with relationship information or None if not a relationship
        """
        if not encoded_path or not isinstance(encoded_path, str):
            return None
            
        if "=" not in encoded_path:
            return None
            
        try:
            # Split into path part and value part
            parts = encoded_path.split("=", 1)
            path_part = parts[0].strip()
            value_part = parts[1].strip()
            
            # Standard relationship pattern: rel:source->target = type
            if "rel:" in path_part and "->" in path_part:
                # Extract after rel: prefix
                rel_def = path_part.replace("rel:", "").strip()
                
                # Handle multiple relationship formats
                if "->" in rel_def:
                    # Standard format: source->target
                    source_path, target_path = rel_def.split("->", 1)
                elif "|" in rel_def:
                    # Alternative format: source|target
                    source_path, target_path = rel_def.split("|", 1)
                elif "=" in rel_def:
                    # Another format sometimes used: source=target
                    source_path, target_path = rel_def.split("=", 1)
                elif ":" in rel_def:
                    # Yet another format: source:target
                    source_path, target_path = rel_def.split(":", 1)
                else:
                    # Fallback: assume the whole thing is source path
                    source_path = rel_def
                    target_path = "values.current"  # Default target for healthcare metrics
                
                # Improve relationship information with enhanced metadata
                rel_info = {
                    "relation_type": value_part,
                    "source_path": source_path.strip(),
                    "target_path": target_path.strip(),
                    "full_encoding": encoded_path,
                    "strength": 0.95,  # Higher confidence for explicit relationships
                    "is_explicit": True,
                    "direction": "source_to_target"
                }
                
                # Additional metadata for healthcare relationships
                if "patient" in value_part.lower() or "satisfaction" in value_part.lower():
                    rel_info["domain"] = "healthcare_satisfaction"
                    rel_info["strength"] = 0.98  # Even higher confidence for healthcare satisfaction
                elif "factor" in value_part.lower() or "influence" in value_part.lower():
                    rel_info["domain"] = "healthcare_factors"
                    rel_info["relationship_category"] = "causal"
                
                return rel_info
            
            # Enhanced detection of implicit healthcare relationships
            healthcare_rel_terms = [
                'influences', 'contributes', 'affects', 'impacts',
                'improvement', 'target_achievement', 'metric_improvement',
                'progress_towards', 'explains', 'provider_communication',
                'wait_time', 'facility_quality', 'staff_training',
                'process_optimization', 'technology_modernization',
                'patient_experience', 'healthcare_quality', 'satisfaction',
                'metric', 'measures', 'defines_goal', 'requires_validation',
                # Enhanced terms for better detection
                'correlation', 'causation', 'indicator', 'predictor',
                'outcome', 'determinant', 'driver', 'leads to', 'results in',
                'patient satisfaction', 'quality measure', 'performance metric',
                'clinical outcome', 'healthcare delivery', 'service quality'
            ]
            
            # Check if value part contains healthcare relation terms
            value_lower = value_part.lower()
            if any(term in value_lower for term in healthcare_rel_terms):
                # Try to extract source/target from context
                if ':' in path_part:
                    marker, path = path_part.split(':', 1)
                    path = path.strip()
                    
                    # Enhanced handling of different marker types
                    if marker.lower() in ('m', 'metric', 'measure', 'measurement'):
                        # This is a metric value relationship
                        if 'values.' in path or 'score' in path or 'rating' in path:
                            # Determine appropriate relationship type and target
                            if 'current' in path or 'latest' in path or 'present' in path:
                                relation_type = "progress_towards_target" if "target" not in value_lower else value_part
                                return {
                                    "relation_type": relation_type,
                                    "source_path": path,
                                    "target_path": "values.target",
                                    "full_encoding": encoded_path,
                                    "strength": 0.92,
                                    "is_explicit": False,
                                    "direction": "source_to_target",
                                    "domain": "healthcare_metrics"
                                }
                            elif 'baseline' in path or 'initial' in path or 'previous' in path:
                                relation_type = "improvement_from_baseline" if "improvement" not in value_lower else value_part
                                return {
                                    "relation_type": relation_type,
                                    "source_path": path,
                                    "target_path": "values.current",
                                    "full_encoding": encoded_path,
                                    "strength": 0.92,
                                    "is_explicit": False,
                                    "direction": "source_to_target",
                                    "domain": "healthcare_metrics"
                                }
                            elif 'target' in path or 'goal' in path:
                                return {
                                    "relation_type": "target_definition",
                                    "source_path": path,
                                    "target_path": "values.current",
                                    "full_encoding": encoded_path,
                                    "strength": 0.9,
                                    "is_explicit": False,
                                    "direction": "target_to_source",
                                    "domain": "healthcare_metrics"
                                }
                    
                    # Enhanced factor and component relationship detection
                    factor_markers = ('f', 'factor', 'influence', 'driver', 'determinant', 'predictor')
                    component_markers = ('c', 'component', 'element', 'part', 'aspect', 'dimension')
                    
                    if any(marker.lower().startswith(m) for m in factor_markers):
                        # Enhanced factor relationship to metric with more metadata
                        rel_type = self._determine_factor_relationship_type(value_lower)
                        return {
                            "relation_type": rel_type,
                            "source_path": path,
                            "target_path": "values.current",
                            "full_encoding": encoded_path,
                            "strength": 0.95,  # Higher confidence
                            "is_explicit": False,
                            "relationship_category": "factor",
                            "direction": "source_to_target",
                            "domain": "healthcare_factors"
                        }
                    elif any(marker.lower().startswith(c) for c in component_markers):
                        # Enhanced component relationship to metric
                        rel_type = self._determine_component_relationship_type(value_lower)
                        return {
                            "relation_type": rel_type,
                            "source_path": path,
                            "target_path": "values.current",
                            "full_encoding": encoded_path,
                            "strength": 0.93,
                            "is_explicit": False,
                            "relationship_category": "component",
                            "direction": "source_to_target",
                            "domain": "healthcare_components"
                        }
            
            # Enhanced detection of patient satisfaction relationships
            if ('patient' in value_lower and 'satisfaction' in value_lower) or 'satisfaction' in value_lower:
                # Try to identify patient satisfaction factors from various formats
                if 'wait' in value_lower or 'time' in value_lower:
                    return {
                        "relation_type": "satisfaction_factor",
                        "source_path": "factors.wait_time",
                        "target_path": "metrics.patient_satisfaction",
                        "full_encoding": encoded_path,
                        "strength": 0.94,
                        "is_explicit": False,
                        "factor_name": "wait_time",
                        "domain": "healthcare_satisfaction"
                    }
                elif 'communication' in value_lower or 'provider' in value_lower:
                    return {
                        "relation_type": "satisfaction_factor",
                        "source_path": "factors.provider_communication",
                        "target_path": "metrics.patient_satisfaction",
                        "full_encoding": encoded_path,
                        "strength": 0.94,
                        "is_explicit": False,
                        "factor_name": "provider_communication",
                        "domain": "healthcare_satisfaction"
                    }
                elif 'facility' in value_lower or 'clean' in value_lower or 'comfort' in value_lower:
                    return {
                        "relation_type": "satisfaction_factor",
                        "source_path": "factors.facility_quality",
                        "target_path": "metrics.patient_satisfaction",
                        "full_encoding": encoded_path,
                        "strength": 0.93,
                        "is_explicit": False,
                        "factor_name": "facility_quality",
                        "domain": "healthcare_satisfaction"
                    }
                elif 'bill' in value_lower or 'cost' in value_lower or 'financ' in value_lower:
                    return {
                        "relation_type": "satisfaction_factor",
                        "source_path": "factors.billing_transparency",
                        "target_path": "metrics.patient_satisfaction",
                        "full_encoding": encoded_path,
                        "strength": 0.92,
                        "is_explicit": False,
                        "factor_name": "billing_transparency",
                        "domain": "healthcare_satisfaction"
                    }
            
            # Nothing detected, return None
            return None
        except Exception as e:
            # Return a minimal relationship for resilience
            return {
                "relation_type": "unknown",
                "source_path": "unknown",
                "target_path": "unknown",
                "full_encoding": encoded_path if isinstance(encoded_path, str) else "invalid",
                "strength": 0.5,
                "is_explicit": False,
                "error": str(e)
            }
            
    def _determine_factor_relationship_type(self, value_text: str) -> str:
        """Determine the specific type of factor relationship based on value text"""
        if 'strong' in value_text or 'significant' in value_text:
            return "strong_factor_influence"
        elif 'moderate' in value_text:
            return "moderate_factor_influence"
        elif 'weak' in value_text or 'minor' in value_text:
            return "weak_factor_influence"
        elif 'negative' in value_text or 'inverse' in value_text:
            return "negative_factor_influence"
        elif 'direct' in value_text:
            return "direct_factor_influence"
        elif 'causal' in value_text:
            return "causal_factor_relationship"
        else:
            return "factor_influences_metric"
            
    def _determine_component_relationship_type(self, value_text: str) -> str:
        """Determine the specific type of component relationship based on value text"""
        if 'key' in value_text or 'critical' in value_text or 'essential' in value_text:
            return "key_component_contribution"
        elif 'supporting' in value_text:
            return "supporting_component_contribution"
        elif 'partial' in value_text:
            return "partial_component_contribution"
        else:
            return "component_contributes_to_metric"
    
    def get_path_info(self, encoded_path: str) -> Dict[str, str]:
        """
        Extract path information from an encoded path string.
        
        Args:
            encoded_path: The encoded path string
            
        Returns:
            Dictionary with path information or None if not properly formatted
        """
        if not encoded_path or "=" not in encoded_path:
            return None
            
        parts = encoded_path.split("=", 1)
        path_part = parts[0].strip()
        value_part = parts[1].strip()
        
        # Extract marker and path
        marker = "p"  # default
        path = path_part
        
        if ":" in path_part:
            marker_part, path = path_part.split(":", 1)
            marker = marker_part
            
        return {
            "marker": marker,
            "path": path.strip(),
            "value": value_part
        }
    
    def calculate_path_encoding_relevance(self, encoded_paths: List[str], 
                                         query: str = None) -> Dict[str, float]:
        """
        Calculate relevance scores for encoded paths.
        
        Args:
            encoded_paths: List of strings with path-based encoding
            query: Optional query to focus relevance calculation
            
        Returns:
            Dictionary mapping paths to relevance scores
        """
        relevance_scores = {}
        
        # Simple implementation using path and value matching
        for path in encoded_paths:
            if "=" not in path:
                continue
                
            parts = path.split("=", 1)
            path_part = parts[0].strip()
            value_part = parts[1].strip()
            
            # Base relevance score
            relevance = 0.5
            
            # Boost relationships
            if "rel:" in path_part:
                relevance += 0.3
            
            # Boost metrics
            if "m:" in path_part:
                relevance += 0.2
                
            # Query relevance if provided
            if query:
                query_lower = query.lower()
                path_lower = path_part.lower()
                value_lower = value_part.lower()
                
                if query_lower in path_lower:
                    relevance += 0.3
                if query_lower in value_lower:
                    relevance += 0.2
                    
            relevance_scores[path] = min(relevance, 1.0)  # Cap at 1.0
            
        return relevance_scores
