"""
Structured comparison method for contrasting actual AI/ML implementations with alternatives.

This module provides tools for systematically comparing what was done with what could
have been done, enabling deeper insights into implementation failures.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

from .causal_analysis import FailureCase, DecisionPoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@dataclass
class ComparisonDimension:
    """A dimension for comparing actual vs. counterfactual decisions."""
    
    name: str
    description: str
    weight: float  # Importance weight (0-1)
    evaluation_criteria: List[str]  # Criteria for evaluating this dimension
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "weight": self.weight,
            "evaluation_criteria": self.evaluation_criteria
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ComparisonDimension':
        """Create from dictionary representation."""
        return cls(
            name=data["name"],
            description=data["description"],
            weight=data["weight"],
            evaluation_criteria=data["evaluation_criteria"]
        )

@dataclass
class ComparisonResult:
    """Result of comparing actual vs. counterfactual decisions."""
    
    dimension: str
    actual_score: float  # 0-10 scale
    counterfactual_score: float  # 0-10 scale
    explanation: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "dimension": self.dimension,
            "actual_score": self.actual_score,
            "counterfactual_score": self.counterfactual_score,
            "explanation": self.explanation
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ComparisonResult':
        """Create from dictionary representation."""
        return cls(
            dimension=data["dimension"],
            actual_score=data["actual_score"],
            counterfactual_score=data["counterfactual_score"],
            explanation=data["explanation"]
        )

@dataclass
class StructuredComparison:
    """A structured comparison between actual and counterfactual scenarios."""
    
    id: str
    failure_case_id: str
    counterfactual_description: str
    dimensions: List[ComparisonDimension]
    results: List[ComparisonResult]
    overall_assessment: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "failure_case_id": self.failure_case_id,
            "counterfactual_description": self.counterfactual_description,
            "dimensions": [d.to_dict() for d in self.dimensions],
            "results": [r.to_dict() for r in self.results],
            "overall_assessment": self.overall_assessment
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StructuredComparison':
        """Create from dictionary representation."""
        return cls(
            id=data["id"],
            failure_case_id=data["failure_case_id"],
            counterfactual_description=data["counterfactual_description"],
            dimensions=[ComparisonDimension.from_dict(d) for d in data["dimensions"]],
            results=[ComparisonResult.from_dict(r) for r in data["results"]],
            overall_assessment=data["overall_assessment"]
        )
    
    def get_weighted_improvement(self) -> float:
        """Calculate the weighted improvement score."""
        dimension_map = {d.name: d.weight for d in self.dimensions}
        
        weighted_sum = 0
        total_weight = 0
        
        for result in self.results:
            weight = dimension_map.get(result.dimension, 1.0)
            improvement = result.counterfactual_score - result.actual_score
            weighted_sum += weight * improvement
            total_weight += weight
        
        if total_weight == 0:
            return 0
        
        return weighted_sum / total_weight

class CounterfactualComparator:
    """Comparator for structured analysis of actual vs. counterfactual scenarios."""
    
    def __init__(self, comparisons_dir: str = "data/counterfactual/comparisons"):
        """
        Initialize the counterfactual comparator.
        
        Args:
            comparisons_dir: Directory for storing comparison data
        """
        self.comparisons_dir = Path(comparisons_dir)
        self.comparisons_dir.mkdir(parents=True, exist_ok=True)
        
        # Standard comparison dimensions
        self.standard_dimensions = [
            ComparisonDimension(
                name="data_quality",
                description="Quality, relevance, and representativeness of the data",
                weight=0.9,
                evaluation_criteria=[
                    "Completeness", "Accuracy", "Relevance", 
                    "Representativeness", "Timeliness"
                ]
            ),
            ComparisonDimension(
                name="model_suitability",
                description="Appropriateness of the model for the specific task",
                weight=0.8,
                evaluation_criteria=[
                    "Task alignment", "Complexity appropriateness", 
                    "Interpretability needs", "Computational efficiency"
                ]
            ),
            ComparisonDimension(
                name="implementation_quality",
                description="Technical quality of the implementation",
                weight=0.7,
                evaluation_criteria=[
                    "Code quality", "Testing coverage", "Documentation",
                    "Maintainability", "Error handling"
                ]
            ),
            ComparisonDimension(
                name="evaluation_rigor",
                description="Rigor and appropriateness of evaluation methods",
                weight=0.8,
                evaluation_criteria=[
                    "Metric selection", "Test/validation approach",
                    "Robustness testing", "Bias assessment"
                ]
            ),
            ComparisonDimension(
                name="deployment_strategy",
                description="Strategy for deployment and ongoing monitoring",
                weight=0.7,
                evaluation_criteria=[
                    "Phased rollout", "Monitoring setup", "Fallback mechanisms",
                    "Update process", "User feedback channels"
                ]
            ),
            ComparisonDimension(
                name="stakeholder_alignment",
                description="Alignment with stakeholder needs and expectations",
                weight=0.9,
                evaluation_criteria=[
                    "User needs addressed", "Business goals alignment",
                    "Expectation management", "Communication clarity"
                ]
            ),
            ComparisonDimension(
                name="ethical_considerations",
                description="Consideration of ethical implications",
                weight=0.8,
                evaluation_criteria=[
                    "Fairness", "Transparency", "Privacy protection",
                    "Societal impact", "Accountability"
                ]
            )
        ]
        
        # Load existing comparisons
        self.comparisons: Dict[str, StructuredComparison] = {}
        self._load_comparisons()
    
    def _load_comparisons(self) -> None:
        """Load comparisons from data directory."""
        comparison_files = list(self.comparisons_dir.glob("*.json"))
        for file_path in comparison_files:
            try:
                with open(file_path, 'r') as f:
                    comparison_data = json.load(f)
                    comparison = StructuredComparison.from_dict(comparison_data)
                    self.comparisons[comparison.id] = comparison
                    logger.info(f"Loaded comparison: {comparison.id}")
            except Exception as e:
                logger.error(f"Error loading comparison from {file_path}: {e}")
        
        logger.info(f"Loaded {len(self.comparisons)} comparisons")
    
    def save_comparison(self, comparison: StructuredComparison) -> None:
        """
        Save a comparison to disk.
        
        Args:
            comparison: The comparison to save
        """
        file_path = self.comparisons_dir / f"{comparison.id}.json"
        with open(file_path, 'w') as f:
            json.dump(comparison.to_dict(), f, indent=2)
        
        self.comparisons[comparison.id] = comparison
        logger.info(f"Saved comparison: {comparison.id}")
        
    def get_comparison(self, comparison_id: str) -> Optional[StructuredComparison]:
        """
        Retrieve a comparison by ID.
        
        Args:
            comparison_id: ID of the comparison to retrieve
            
        Returns:
            The requested comparison, or None if not found
        """
        if comparison_id not in self.comparisons:
            logger.error(f"Comparison not found: {comparison_id}")
            return None
            
        return self.comparisons[comparison_id]
    
    def create_comparison(
        self,
        failure_case: FailureCase,
        counterfactual_description: str,
        dimensions: Optional[List[ComparisonDimension]] = None,
        comparison_id: Optional[str] = None
    ) -> StructuredComparison:
        """
        Create a new structured comparison.
        
        Args:
            failure_case: The failure case to analyze
            counterfactual_description: Description of the counterfactual scenario
            dimensions: Comparison dimensions, or None to use standard dimensions
            comparison_id: Optional ID for the comparison
            
        Returns:
            A new structured comparison
        """
        if dimensions is None:
            dimensions = self.standard_dimensions
        
        if comparison_id is None:
            comparison_id = f"comp_{failure_case.id}_{len(self.comparisons)}"
        
        # Create empty comparison
        comparison = StructuredComparison(
            id=comparison_id,
            failure_case_id=failure_case.id,
            counterfactual_description=counterfactual_description,
            dimensions=dimensions,
            results=[],
            overall_assessment=""
        )
        
        return comparison
    
    def add_comparison_result(
        self,
        comparison: StructuredComparison,
        dimension: str,
        actual_score: float,
        counterfactual_score: float,
        explanation: str
    ) -> None:
        """
        Add a result to a comparison.
        
        Args:
            comparison: The comparison to update
            dimension: Name of the comparison dimension
            actual_score: Score for the actual implementation (0-10)
            counterfactual_score: Score for the counterfactual scenario (0-10)
            explanation: Explanation of the scores
        """
        result = ComparisonResult(
            dimension=dimension,
            actual_score=actual_score,
            counterfactual_score=counterfactual_score,
            explanation=explanation
        )
        
        # Check if result for this dimension already exists
        for i, existing_result in enumerate(comparison.results):
            if existing_result.dimension == dimension:
                comparison.results[i] = result
                return
        
        comparison.results.append(result)
    
    def set_overall_assessment(self, comparison: StructuredComparison, assessment: str) -> None:
        """
        Set the overall assessment for a comparison.
        
        Args:
            comparison: The comparison to update
            assessment: Overall assessment text
        """
        comparison.overall_assessment = assessment
    
    def visualize_comparison(
        self, 
        comparison: StructuredComparison,
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a visualization of the comparison.
        
        Args:
            comparison: The comparison to visualize
            output_path: Path to save visualization, or None to not save
            
        Returns:
            Matplotlib figure
        """
        # Prepare data
        dimensions = [r.dimension for r in comparison.results]
        actual_scores = [r.actual_score for r in comparison.results]
        counterfactual_scores = [r.counterfactual_score for r in comparison.results]
        
        df = pd.DataFrame({
            'Dimension': dimensions,
            'Actual': actual_scores,
            'Counterfactual': counterfactual_scores
        })
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = range(len(dimensions))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], actual_scores, width, label='Actual')
        ax.bar([i + width/2 for i in x], counterfactual_scores, width, label='Counterfactual')
        
        ax.set_ylabel('Score (0-10)')
        ax.set_title('Actual vs. Counterfactual Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(dimensions, rotation=45, ha='right')
        ax.legend()
        
        fig.tight_layout()
        
        # Save if output path provided
        if output_path:
            fig.savefig(output_path)
            logger.info(f"Saved visualization to {output_path}")
        
        return fig
    
    def generate_comparison_report(self, comparison: StructuredComparison) -> str:
        """
        Generate a textual report of the comparison.
        
        Args:
            comparison: The comparison to report on
            
        Returns:
            Formatted report text
        """
        # Calculate weighted improvement
        weighted_improvement = comparison.get_weighted_improvement()
        
        # Generate report
        report = f"# Counterfactual Analysis Report\n\n"
        report += f"## Failure Case: {comparison.failure_case_id}\n\n"
        report += f"### Counterfactual Scenario\n{comparison.counterfactual_description}\n\n"
        
        report += f"### Dimension Comparison\n\n"
        report += "| Dimension | Actual | Counterfactual | Improvement | Explanation |\n"
        report += "|-----------|--------|---------------|-------------|-------------|\n"
        
        for result in comparison.results:
            improvement = result.counterfactual_score - result.actual_score
            report += f"| {result.dimension} | {result.actual_score:.1f} | {result.counterfactual_score:.1f} | {improvement:+.1f} | {result.explanation} |\n"
        
        report += f"\n### Overall Assessment\n\n"
        report += f"Weighted improvement score: {weighted_improvement:+.2f}\n\n"
        report += comparison.overall_assessment
        
        return report
