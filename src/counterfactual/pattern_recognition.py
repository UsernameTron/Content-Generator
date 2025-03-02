"""
Pattern recognition for identifying recurring failure patterns across different sectors.

This module provides tools for analyzing and identifying common patterns of failure
in AI/ML implementations across different industries and project types.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
import uuid

from .causal_analysis import FailureCase, DecisionPoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@dataclass
class FailurePattern:
    """A pattern of failures identified across multiple cases."""
    
    id: str
    name: str
    description: str
    failure_modes: List[str]  # Common failure modes in this pattern
    decision_patterns: List[Dict]  # Common decision patterns
    affected_industries: List[str]  # Industries where this pattern occurs
    affected_project_types: List[str]  # Project types where this pattern occurs
    case_ids: List[str]  # IDs of cases exhibiting this pattern
    severity: float  # Average severity (0-10) of failures in this pattern
    frequency: float  # Frequency of occurrence (0-1)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "failure_modes": self.failure_modes,
            "decision_patterns": self.decision_patterns,
            "affected_industries": self.affected_industries,
            "affected_project_types": self.affected_project_types,
            "case_ids": self.case_ids,
            "severity": self.severity,
            "frequency": self.frequency
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FailurePattern':
        """Create from dictionary representation."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            failure_modes=data["failure_modes"],
            decision_patterns=data["decision_patterns"],
            affected_industries=data["affected_industries"],
            affected_project_types=data["affected_project_types"],
            case_ids=data["case_ids"],
            severity=data["severity"],
            frequency=data["frequency"]
        )

class PatternRecognizer:
    """Recognizer for identifying patterns in AI/ML implementation failures."""
    
    def __init__(
        self, 
        patterns_dir: str = "data/counterfactual/patterns",
        cases_dir: str = "data/counterfactual/failure_cases"
    ):
        """
        Initialize the pattern recognizer.
        
        Args:
            patterns_dir: Directory for storing pattern data
            cases_dir: Directory containing failure case data
        """
        self.patterns_dir = Path(patterns_dir)
        self.patterns_dir.mkdir(parents=True, exist_ok=True)
        
        self.cases_dir = Path(cases_dir)
        
        # Load existing patterns
        self.patterns: Dict[str, FailurePattern] = {}
        self._load_patterns()
        
        # Load failure cases
        self.failure_cases: Dict[str, FailureCase] = {}
        self._load_cases()
        
        # Pattern identification thresholds
        self.similarity_threshold = 0.7  # Minimum similarity to group cases
        self.min_cases_for_pattern = 2  # Minimum cases to form a pattern
    
    def _load_patterns(self) -> None:
        """Load patterns from data directory."""
        pattern_files = list(self.patterns_dir.glob("*.json"))
        for file_path in pattern_files:
            try:
                with open(file_path, 'r') as f:
                    pattern_data = json.load(f)
                    pattern = FailurePattern.from_dict(pattern_data)
                    self.patterns[pattern.id] = pattern
                    logger.info(f"Loaded pattern: {pattern.id} - {pattern.name}")
            except Exception as e:
                logger.error(f"Error loading pattern from {file_path}: {e}")
        
        logger.info(f"Loaded {len(self.patterns)} patterns")
    
    def _load_cases(self) -> None:
        """Load failure cases from data directory."""
        case_files = list(self.cases_dir.glob("*.json"))
        for file_path in case_files:
            try:
                with open(file_path, 'r') as f:
                    case_data = json.load(f)
                    case = FailureCase.from_dict(case_data)
                    self.failure_cases[case.id] = case
                    logger.info(f"Loaded failure case: {case.id} - {case.title}")
            except Exception as e:
                logger.error(f"Error loading failure case from {file_path}: {e}")
        
        logger.info(f"Loaded {len(self.failure_cases)} failure cases")
    
    def add_failure_case(self, failure_case: FailureCase) -> None:
        """
        Add a failure case to the pattern recognizer.
        
        Args:
            failure_case: The failure case to add
        """
        # Store the case in memory
        self.failure_cases[failure_case.id] = failure_case
        
        # Save the case to disk
        file_path = self.cases_dir / f"{failure_case.id}.json"
        try:
            with open(file_path, 'w') as f:
                json.dump(failure_case.to_dict(), f, indent=2)
            logger.info(f"Added failure case: {failure_case.id} - {failure_case.title}")
        except Exception as e:
            logger.error(f"Error saving failure case {failure_case.id}: {e}")
    
    def save_pattern(self, pattern: FailurePattern) -> None:
        """
        Save a pattern to disk.
        
        Args:
            pattern: The pattern to save
        """
        file_path = self.patterns_dir / f"{pattern.id}.json"
        with open(file_path, 'w') as f:
            json.dump(pattern.to_dict(), f, indent=2)
        
        self.patterns[pattern.id] = pattern
        logger.info(f"Saved pattern: {pattern.id} - {pattern.name}")
    
    def identify_patterns(self) -> List[FailurePattern]:
        """
        Identify patterns across failure cases.
        
        Returns:
            List of identified patterns
        """
        # Check if we have enough cases to identify patterns
        if len(self.failure_cases) < self.min_cases_for_pattern:
            logger.warning(f"Not enough failure cases to identify patterns (minimum {self.min_cases_for_pattern})")
            return []
        
        # Group cases by similar failure modes
        failure_mode_groups = self._group_by_failure_modes()
        
        # Create patterns from groups
        patterns = []
        for i, (modes, cases) in enumerate(failure_mode_groups.items()):
            if len(cases) < self.min_cases_for_pattern:
                continue
            
            pattern_id = f"pattern_{i+1}"
            
            # Extract common attributes
            industries = self._extract_common_industries(cases)
            project_types = self._extract_common_project_types(cases)
            decision_patterns = self._extract_decision_patterns(cases)
            
            # Calculate severity (arbitrarily assigned for now)
            severity = 7.5  # On a scale of 0-10
            
            # Calculate frequency
            frequency = len(cases) / len(self.failure_cases)
            
            # Create pattern name and description
            name = f"Pattern: {' & '.join(modes[:2])}"
            description = f"A pattern of AI/ML implementation failures characterized by {', '.join(modes)}."
            
            # Create pattern
            pattern = FailurePattern(
                id=pattern_id,
                name=name,
                description=description,
                failure_modes=modes.split(","),
                decision_patterns=decision_patterns,
                affected_industries=industries,
                affected_project_types=project_types,
                case_ids=[case.id for case in cases],
                severity=severity,
                frequency=frequency
            )
            
            patterns.append(pattern)
        
        logger.info(f"Identified {len(patterns)} patterns across {len(self.failure_cases)} cases")
        return patterns
    
    def _group_by_failure_modes(self) -> Dict[str, List[FailureCase]]:
        """
        Group failure cases by similar failure modes.
        
        Returns:
            Dictionary mapping failure mode combinations to lists of cases
        """
        # Extract all unique failure modes
        all_modes = set()
        for case in self.failure_cases.values():
            all_modes.update(case.primary_failure_modes)
        
        # Group cases by shared failure modes
        groups = {}
        
        for case in self.failure_cases.values():
            # Create a key from sorted failure modes
            mode_key = ",".join(sorted(case.primary_failure_modes))
            
            if mode_key not in groups:
                groups[mode_key] = []
            
            groups[mode_key].append(case)
        
        return groups
    
    def _extract_common_industries(self, cases: List[FailureCase]) -> List[str]:
        """Extract common industries from a list of cases."""
        industries = [case.industry for case in cases]
        counter = Counter(industries)
        
        # Include industries that appear in at least 30% of cases
        threshold = max(1, len(cases) * 0.3)
        common_industries = [industry for industry, count in counter.items() if count >= threshold]
        
        return common_industries
    
    def _extract_common_project_types(self, cases: List[FailureCase]) -> List[str]:
        """Extract common project types from a list of cases."""
        project_types = [case.project_type for case in cases]
        counter = Counter(project_types)
        
        # Include project types that appear in at least 30% of cases
        threshold = max(1, len(cases) * 0.3)
        common_types = [ptype for ptype, count in counter.items() if count >= threshold]
        
        return common_types
    
    def _extract_decision_patterns(self, cases: List[FailureCase]) -> List[Dict]:
        """Extract common decision patterns from a list of cases."""
        # Group decision points by stage
        stage_decisions = {}
        
        for case in cases:
            for dp in case.decision_points:
                if dp.stage not in stage_decisions:
                    stage_decisions[dp.stage] = []
                
                stage_decisions[dp.stage].append({
                    "case_id": case.id,
                    "decision": dp.actual_decision,
                    "description": dp.description,
                    "importance": dp.importance
                })
        
        # Find common decisions within each stage
        common_patterns = []
        
        for stage, decisions in stage_decisions.items():
            # Group similar decisions
            decision_groups = {}
            
            for decision in decisions:
                # Simplified grouping by decision value
                key = decision["decision"]
                
                if key not in decision_groups:
                    decision_groups[key] = []
                
                decision_groups[key].append(decision)
            
            # Find frequent decisions
            for decision_key, group in decision_groups.items():
                if len(group) >= self.min_cases_for_pattern:
                    common_patterns.append({
                        "stage": stage,
                        "decision": decision_key,
                        "description": group[0]["description"],
                        "frequency": len(group) / len(cases),
                        "cases": [d["case_id"] for d in group]
                    })
        
        return common_patterns
    
    def get_pattern_by_case(self, case_id: str) -> List[FailurePattern]:
        """
        Get patterns that include a specific case.
        
        Args:
            case_id: ID of the failure case
            
        Returns:
            List of patterns that include this case
        """
        matching_patterns = []
        
        for pattern in self.patterns.values():
            if case_id in pattern.case_ids:
                matching_patterns.append(pattern)
        
        return matching_patterns
    
    def identify_patterns_in_case(self, case_id: str) -> List[FailurePattern]:
        """
        Identify patterns in a specific case, even if not previously categorized.
        
        This method attempts to match a case with existing patterns and also
        dynamically identifies new patterns by comparing with other cases.
        
        Args:
            case_id: ID of the case to analyze
            
        Returns:
            List of pattern objects that match this case
        """
        logger.info(f"Identifying patterns for case: {case_id}")
        
        if case_id not in self.failure_cases:
            logger.error(f"Failure case not found: {case_id}")
            return []
            
        # Get the case
        case = self.failure_cases[case_id]
        
        # Find similar cases
        similar_cases = self._find_similar_cases(case)
        
        if not similar_cases:
            logger.info(f"No similar cases found for {case_id}")
            
            # If no similar cases found, create a new pattern based on this case alone
            decision_types = set()
            for dp in case.decision_points:
                # Extract a decision type from the description - simplified version
                words = dp.description.lower().split()
                key_type_words = ["data", "model", "eval", "deploy", "monitor", "validate"]
                
                for word in key_type_words:
                    if any(word in w for w in words):
                        decision_types.add(word)
                        break
            
            # Create pattern name based on decision types
            if decision_types:
                pattern_name = f"{'-'.join(sorted(decision_types))}_related_failure"
            else:
                pattern_name = "general_implementation_failure"
                
            # Create a new pattern
            pattern = FailurePattern(
                id=f"pattern_{case_id}",
                name=pattern_name,
                description=f"Pattern derived from analysis of {case.title}",
                failure_modes=[f"Decisions related to {mode}" for mode in decision_types] if decision_types else ["Various implementation decisions"],
                decision_patterns=[],
                affected_industries=[case.industry],
                affected_project_types=[case.project_type],
                case_ids=[case_id],
                severity=7.0,
                frequency=1
            )
            
            # Save the pattern
            self.patterns[pattern.id] = pattern
            self.save_pattern(pattern)
            
            return [pattern]
            
        # Check existing patterns
        matching_patterns = []
        for pattern in self.patterns.values():
            # Check if case is already in this pattern
            if case_id in pattern.case_ids:
                matching_patterns.append(pattern)
                continue
                
            # Check if similar cases are in this pattern
            if any(similar_id in pattern.case_ids for similar_id in [c.id for c in similar_cases]):
                # Add this case to the pattern
                pattern.case_ids.append(case_id)
                pattern.frequency += 1
                self.save_pattern(pattern)
                matching_patterns.append(pattern)
                
        # If no matches in existing patterns, create a new one
        if not matching_patterns:
            # Group the case with its similar cases
            case_group = [case] + similar_cases
            
            # Analyze common decision points across cases
            common_factors = self._identify_common_factors(case_group)
            
            # Create a pattern name based on common factors
            if common_factors:
                pattern_name = f"{'-'.join(sorted(common_factors))}_related_failure"
            else:
                pattern_name = "implementation_failure_pattern"
                
            # Create new pattern
            pattern = FailurePattern(
                id=f"pattern_{uuid.uuid4().hex[:8]}",
                name=pattern_name,
                description=f"Pattern identified across {len(case_group)} similar failure cases",
                failure_modes=[f"Decisions related to {factor}" for factor in common_factors] if common_factors else ["Multiple implementation decisions"],
                decision_patterns=self._extract_decision_patterns(case_group),
                affected_industries=self._extract_common_industries(case_group),
                affected_project_types=self._extract_common_project_types(case_group),
                case_ids=[c.id for c in case_group],
                severity=7.0,
                frequency=len(case_group)
            )
            
            # Save the pattern
            self.patterns[pattern.id] = pattern
            self.save_pattern(pattern)
            matching_patterns.append(pattern)
            
        return matching_patterns
    
    def _find_similar_cases(self, case: FailureCase) -> List[FailureCase]:
        """Find cases similar to the given case."""
        similar_cases = []
        
        for other_id, other_case in self.failure_cases.items():
            # Skip comparing with self
            if other_id == case.id:
                continue
                
            # Check for similarity in failure modes
            mode_similarity = self._calculate_failure_mode_similarity(case, other_case)
            
            # Check for similarity in industry/project type
            domain_similarity = 0.0
            if case.industry == other_case.industry:
                domain_similarity += 0.5
            if case.project_type == other_case.project_type:
                domain_similarity += 0.5
                
            # Calculate overall similarity score
            similarity = (mode_similarity * 0.7) + (domain_similarity * 0.3)
            
            # Add to similar cases if above threshold
            if similarity >= self.similarity_threshold:
                similar_cases.append(other_case)
                
        return similar_cases
        
    def _calculate_failure_mode_similarity(self, case1: FailureCase, case2: FailureCase) -> float:
        """Calculate similarity between failure modes of two cases."""
        modes1 = set(case1.primary_failure_modes)
        modes2 = set(case2.primary_failure_modes)
        
        # Calculate Jaccard similarity
        intersection = len(modes1.intersection(modes2))
        union = len(modes1.union(modes2))
        
        return intersection / max(1, union)
    
    def _identify_common_factors(self, cases: List[FailureCase]) -> List[str]:
        """
        Identify common factors across a group of failure cases.
        
        Args:
            cases: List of failure cases to analyze
            
        Returns:
            List of common factors identified across cases
        """
        # Extract words from decision point descriptions
        all_words = []
        for case in cases:
            for dp in case.decision_points:
                words = [w.lower() for w in dp.description.split() if len(w) > 3]
                all_words.extend(words)
                
        # Count word frequencies
        word_counts = Counter(all_words)
        
        # Filter to get meaningful common factors
        key_type_words = ["data", "model", "evaluation", "validation", "deployment", "monitoring"]
        common_factors = []
        
        for word in key_type_words:
            if word in word_counts and word_counts[word] >= (len(cases) / 2):
                common_factors.append(word)
                
        return common_factors
    
    def visualize_pattern_network(self, output_path: Optional[str] = None) -> plt.Figure:
        """
        Create a network visualization of patterns and cases.
        
        Args:
            output_path: Path to save visualization, or None to not save
            
        Returns:
            Matplotlib figure
        """
        # Create graph
        G = nx.Graph()
        
        # Add pattern nodes
        for pattern_id, pattern in self.patterns.items():
            G.add_node(pattern_id, 
                      type="pattern", 
                      name=pattern.name,
                      size=50 * pattern.frequency)
        
        # Add case nodes and edges
        for case_id, case in self.failure_cases.items():
            G.add_node(case_id, 
                      type="case", 
                      name=case.title,
                      size=20)
            
            # Connect to patterns
            for pattern_id, pattern in self.patterns.items():
                if case_id in pattern.case_ids:
                    G.add_edge(pattern_id, case_id)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Set positions using spring layout
        pos = nx.spring_layout(G)
        
        # Get node attributes
        node_types = nx.get_node_attributes(G, "type")
        node_sizes = nx.get_node_attributes(G, "size")
        
        # Draw pattern nodes
        pattern_nodes = [node for node, type_val in node_types.items() if type_val == "pattern"]
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=pattern_nodes,
                              node_size=[node_sizes.get(n, 50) for n in pattern_nodes],
                              node_color="red",
                              alpha=0.8)
        
        # Draw case nodes
        case_nodes = [node for node, type_val in node_types.items() if type_val == "case"]
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=case_nodes,
                              node_size=[node_sizes.get(n, 20) for n in case_nodes],
                              node_color="blue",
                              alpha=0.6)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
        
        # Draw labels for patterns only (to avoid clutter)
        pattern_labels = {node: node for node in pattern_nodes}
        nx.draw_networkx_labels(G, pos, labels=pattern_labels, font_size=8)
        
        plt.title("Failure Patterns Network")
        plt.axis("off")
        
        # Save if output path provided
        if output_path:
            fig.savefig(output_path)
            logger.info(f"Saved visualization to {output_path}")
        
        return fig
    
    def generate_pattern_report(self, pattern_id: str) -> str:
        """
        Generate a textual report for a pattern.
        
        Args:
            pattern_id: ID of the pattern
            
        Returns:
            Formatted report text
        """
        if pattern_id not in self.patterns:
            logger.error(f"Pattern not found: {pattern_id}")
            return ""
        
        pattern = self.patterns[pattern_id]
        
        # Generate report
        report = f"# AI/ML Implementation Failure Pattern Report\n\n"
        report += f"## Pattern: {pattern.name}\n\n"
        report += f"{pattern.description}\n\n"
        
        report += f"### Key Characteristics\n\n"
        report += f"- **Severity:** {pattern.severity:.1f}/10\n"
        report += f"- **Frequency:** {pattern.frequency:.1%} of analyzed cases\n"
        report += f"- **Industries Affected:** {', '.join(pattern.affected_industries)}\n"
        report += f"- **Project Types Affected:** {', '.join(pattern.affected_project_types)}\n\n"
        
        report += f"### Common Failure Modes\n\n"
        for mode in pattern.failure_modes:
            report += f"- {mode}\n"
        
        report += f"\n### Common Decision Patterns\n\n"
        for dp in pattern.decision_patterns:
            report += f"- **{dp['stage']} stage:** {dp['description']} → *{dp['decision']}* ({dp['frequency']:.1%} of cases)\n"
        
        report += f"\n### Cases Exhibiting This Pattern\n\n"
        for case_id in pattern.case_ids:
            if case_id in self.failure_cases:
                case = self.failure_cases[case_id]
                report += f"- {case.title} ({case.industry}, {case.project_type})\n"
        
        return report
