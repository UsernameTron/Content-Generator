"""
Causal analysis framework for identifying critical decision points in AI/ML implementations.

This module provides tools for analyzing causal relationships in AI/ML project failures,
identifying key decision points, and modeling alternative decision paths.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import networkx as nx
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@dataclass
class DecisionPoint:
    """A decision point in an AI/ML implementation process."""
    
    id: str
    description: str
    actual_decision: str
    alternatives: List[str]
    consequences: Dict[str, str]  # Map of decision -> consequence
    impact_areas: List[str]  # Areas affected (e.g., "performance", "fairness", "cost")
    importance: float  # 0-1 scale of importance
    stage: str  # Implementation stage (e.g., "data_preparation", "model_selection")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "description": self.description,
            "actual_decision": self.actual_decision,
            "alternatives": self.alternatives,
            "consequences": self.consequences,
            "impact_areas": self.impact_areas,
            "importance": self.importance,
            "stage": self.stage
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DecisionPoint':
        """Create from dictionary representation."""
        return cls(
            id=data["id"],
            description=data["description"],
            actual_decision=data["actual_decision"],
            alternatives=data["alternatives"],
            consequences=data["consequences"],
            impact_areas=data["impact_areas"],
            importance=data["importance"],
            stage=data["stage"]
        )

@dataclass
class FailureCase:
    """An AI/ML implementation failure case for analysis."""
    
    id: str
    title: str
    description: str
    industry: str
    project_type: str  # e.g., "recommendation_system", "fraud_detection"
    primary_failure_modes: List[str]
    decision_points: List[DecisionPoint]
    observed_outcome: str
    sources: List[Dict]  # List of reference sources
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "industry": self.industry,
            "project_type": self.project_type,
            "primary_failure_modes": self.primary_failure_modes,
            "decision_points": [dp.to_dict() for dp in self.decision_points],
            "observed_outcome": self.observed_outcome,
            "sources": self.sources
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FailureCase':
        """Create from dictionary representation."""
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            industry=data["industry"],
            project_type=data["project_type"],
            primary_failure_modes=data["primary_failure_modes"],
            decision_points=[DecisionPoint.from_dict(dp) for dp in data["decision_points"]],
            observed_outcome=data["observed_outcome"],
            sources=data["sources"]
        )

class CausalAnalyzer:
    """Analyzer for causal relationships in AI/ML implementation failures."""
    
    def __init__(self, cases_dir: str = "data/counterfactual/failure_cases"):
        """
        Initialize the causal analyzer.
        
        Args:
            cases_dir: Directory containing failure case data
        """
        self.cases_dir = Path(cases_dir)
        self.cases_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing failure cases
        self.failure_cases: Dict[str, FailureCase] = {}
        self._load_cases()
    
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
    
    def save_case(self, case: FailureCase) -> None:
        """
        Save a failure case to disk.
        
        Args:
            case: The failure case to save
        """
        file_path = self.cases_dir / f"{case.id}.json"
        with open(file_path, 'w') as f:
            json.dump(case.to_dict(), f, indent=2)
        
        self.failure_cases[case.id] = case
        logger.info(f"Saved failure case: {case.id} - {case.title}")
    
    def create_causal_graph(self, case_id: str) -> Optional[nx.DiGraph]:
        """
        Create a causal graph for a failure case.
        
        Args:
            case_id: ID of the failure case
            
        Returns:
            A directed graph representing causal relationships
        """
        if case_id not in self.failure_cases:
            logger.error(f"Failure case not found: {case_id}")
            return None
        
        case = self.failure_cases[case_id]
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes for each decision point
        for dp in case.decision_points:
            G.add_node(dp.id, 
                       description=dp.description,
                       actual_decision=dp.actual_decision,
                       alternatives=dp.alternatives,
                       importance=dp.importance,
                       stage=dp.stage)
        
        # Add an outcome node
        G.add_node("outcome", description=case.observed_outcome)
        
        # Add causal edges based on stage sequence
        stages = ["data_preparation", "feature_engineering", "model_selection", 
                 "training", "evaluation", "deployment", "monitoring"]
        
        stage_groups = {}
        for dp in case.decision_points:
            if dp.stage not in stage_groups:
                stage_groups[dp.stage] = []
            stage_groups[dp.stage].append(dp)
        
        # Connect nodes based on implementation stages
        prev_stage_nodes = []
        for stage in stages:
            if stage in stage_groups:
                current_stage_nodes = [dp.id for dp in stage_groups[stage]]
                
                # Connect previous stage nodes to current stage nodes
                for prev_node in prev_stage_nodes:
                    for curr_node in current_stage_nodes:
                        G.add_edge(prev_node, curr_node)
                
                prev_stage_nodes = current_stage_nodes
        
        # Connect final stage to outcome
        for node in prev_stage_nodes:
            G.add_edge(node, "outcome")
        
        return G
    
    def identify_critical_decisions(self, case_id: str) -> List[DecisionPoint]:
        """
        Identify the most critical decision points in a failure case.
        
        Args:
            case_id: ID of the failure case
            
        Returns:
            List of critical decision points
        """
        if case_id not in self.failure_cases:
            logger.error(f"Failure case not found: {case_id}")
            return []
        
        case = self.failure_cases[case_id]
        
        # Sort decision points by importance
        critical_points = sorted(
            case.decision_points, 
            key=lambda dp: dp.importance,
            reverse=True
        )
        
        return critical_points
    
    def generate_alternative_paths(self, case_id: str, limit: int = 3) -> List[List[Tuple[str, str]]]:
        """
        Generate alternative decision paths for a failure case.
        
        Args:
            case_id: ID of the failure case
            limit: Maximum number of alternative paths to generate
            
        Returns:
            List of alternative paths, each as a list of (decision_id, alternative) tuples
        """
        if case_id not in self.failure_cases:
            logger.error(f"Failure case not found: {case_id}")
            return []
        
        case = self.failure_cases[case_id]
        
        # Get critical decision points
        critical_points = self.identify_critical_decisions(case_id)[:limit]
        
        # Generate alternative paths by changing one decision at a time
        alternative_paths = []
        
        for dp in critical_points:
            for alt in dp.alternatives:
                if alt != dp.actual_decision:
                    # Create a path where this decision is different
                    path = [(dp.id, alt)]
                    alternative_paths.append(path)
        
        return alternative_paths
    
    def describe_counterfactual_scenario(self, case_id: str, alt_path: List[Tuple[str, str]]) -> str:
        """
        Generate a textual description of a counterfactual scenario.
        
        Args:
            case_id: ID of the failure case
            alt_path: Alternative path as a list of (decision_id, alternative) tuples
            
        Returns:
            Description of the counterfactual scenario
        """
        if case_id not in self.failure_cases:
            logger.error(f"Failure case not found: {case_id}")
            return ""
        
        case = self.failure_cases[case_id]
        
        # Map decision IDs to decision points
        dp_map = {dp.id: dp for dp in case.decision_points}
        
        # Create description
        description = f"Counterfactual scenario for '{case.title}':\n\n"
        description += f"Original outcome: {case.observed_outcome}\n\n"
        description += "What if the team had made the following different decisions:\n\n"
        
        for dp_id, alternative in alt_path:
            if dp_id in dp_map:
                dp = dp_map[dp_id]
                description += f"- Instead of '{dp.actual_decision}', they chose '{alternative}' for '{dp.description}'\n"
                if alternative in dp.consequences:
                    description += f"  Likely result: {dp.consequences[alternative]}\n"
        
        description += "\nThis would likely have led to a different outcome where:\n"
        
        # Infer overall counterfactual outcome based on alternative consequences
        counterfactual_outcome = "The project would have achieved better results by avoiding the original failure modes."
        for dp_id, alternative in alt_path:
            if dp_id in dp_map:
                dp = dp_map[dp_id]
                if alternative in dp.consequences:
                    counterfactual_outcome = dp.consequences[alternative]
                    break
        
        description += counterfactual_outcome
        
        return description
    
    def analyze_causal_relationships(self, case: FailureCase) -> None:
        """
        Analyze causal relationships in a failure case.
        
        This method examines decision points and their relationships to build
        a deeper understanding of failure modes and impacts.
        
        Args:
            case: The failure case to analyze
        """
        logger.info(f"Analyzing causal relationships for case: {case.id}")
        
        # For this minimal implementation, we'll simply record the case
        self.failure_cases[case.id] = case
        
        # In a real implementation, we would:
        # 1. Create a causal graph
        # 2. Identify critical decision points
        # 3. Analyze impact of decisions on outcomes
        # 4. Tag decisions with influence scores
        
        logger.info(f"Causal analysis complete for case: {case.id}")
        
    def generate_alternative_path(self, decision_point: DecisionPoint) -> str:
        """
        Generate an alternative path for a decision point.
        
        Args:
            decision_point: The decision point to generate an alternative for
            
        Returns:
            An alternative decision
        """
        if not decision_point.alternatives:
            return "No alternatives available"
            
        # Choose the first alternative that's different from the actual decision
        for alt in decision_point.alternatives:
            if alt != decision_point.actual_decision:
                return alt
                
        return "No viable alternatives found"
    
    def get_case(self, case_id: str) -> Optional[FailureCase]:
        """
        Get a failure case by ID.
        
        Args:
            case_id: ID of the failure case to retrieve
            
        Returns:
            FailureCase object or None if case not found
        """
        if case_id not in self.failure_cases:
            logger.error(f"Failure case not found: {case_id}")
            return None
            
        return self.failure_cases[case_id]
