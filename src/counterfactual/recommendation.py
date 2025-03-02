"""
Practical recommendation generation based on counterfactual reasoning insights.

This module provides tools for generating actionable recommendations based on
counterfactual analysis of AI/ML implementation failures.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
from pathlib import Path

from .causal_analysis import FailureCase, DecisionPoint
from .comparison import StructuredComparison

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@dataclass
class Recommendation:
    """A practical recommendation derived from counterfactual analysis."""
    
    id: str
    title: str
    description: str
    source_comparison_id: str
    priority: int  # 1 (highest) to 5 (lowest)
    difficulty: int  # 1 (easiest) to 5 (hardest)
    impact: int  # 1 (lowest) to 5 (highest)
    applicable_stages: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "source_comparison_id": self.source_comparison_id,
            "priority": self.priority,
            "difficulty": self.difficulty,
            "impact": self.impact,
            "applicable_stages": self.applicable_stages
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Recommendation':
        """Create from dictionary representation."""
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            source_comparison_id=data["source_comparison_id"],
            priority=data["priority"],
            difficulty=data["difficulty"],
            impact=data["impact"],
            applicable_stages=data["applicable_stages"]
        )
    
    @property
    def priority_impact_ratio(self) -> float:
        """Calculate priority-to-impact ratio (lower is better)."""
        if self.impact == 0:
            return float('inf')
        return self.priority / self.impact

class RecommendationGenerator:
    """Generator for practical recommendations from counterfactual analysis."""
    
    def __init__(self, recommendations_dir: str = "data/counterfactual/recommendations"):
        """
        Initialize the recommendation generator.
        
        Args:
            recommendations_dir: Directory for storing recommendation data
        """
        self.recommendations_dir = Path(recommendations_dir)
        self.recommendations_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing recommendations
        self.recommendations: Dict[str, Recommendation] = {}
        self._load_recommendations()
    
    def _load_recommendations(self) -> None:
        """Load recommendations from data directory."""
        recommendation_files = list(self.recommendations_dir.glob("*.json"))
        for file_path in recommendation_files:
            try:
                with open(file_path, 'r') as f:
                    recommendation_data = json.load(f)
                    recommendation = Recommendation.from_dict(recommendation_data)
                    self.recommendations[recommendation.id] = recommendation
                    logger.info(f"Loaded recommendation: {recommendation.id} - {recommendation.title}")
            except Exception as e:
                logger.error(f"Error loading recommendation from {file_path}: {e}")
        
        logger.info(f"Loaded {len(self.recommendations)} recommendations")
    
    def save_recommendation(self, recommendation: Recommendation) -> None:
        """
        Save a recommendation to disk.
        
        Args:
            recommendation: The recommendation to save
        """
        file_path = self.recommendations_dir / f"{recommendation.id}.json"
        with open(file_path, 'w') as f:
            json.dump(recommendation.to_dict(), f, indent=2)
        
        self.recommendations[recommendation.id] = recommendation
        logger.info(f"Saved recommendation: {recommendation.id} - {recommendation.title}")
    
    def generate_recommendations(self, comparison: StructuredComparison) -> List[Recommendation]:
        """
        Generate practical recommendations from a counterfactual comparison.
        
        Args:
            comparison: The comparison to generate recommendations from
            
        Returns:
            List of practical recommendations
        """
        recommendations = []
        
        # Extract key insights from comparison results
        for i, result in enumerate(comparison.results):
            # Skip if no significant improvement in counterfactual scenario
            if result.counterfactual_score - result.actual_score < 1.0:
                continue
            
            # Create recommendation
            rec_id = f"rec_{comparison.id}_{i+1}"
            
            # Derive title from dimension
            title = f"Improve {result.dimension.replace('_', ' ')} in AI/ML implementation"
            
            # Derive description from improvement explanation
            description = (
                f"Based on counterfactual analysis, improving {result.dimension.replace('_', ' ')} "
                f"could significantly enhance outcomes. {result.explanation}\n\n"
                f"The counterfactual scenario demonstrated a {result.counterfactual_score - result.actual_score:.1f} "
                f"point improvement (on a 10-point scale) in this dimension."
            )
            
            # Set priority based on improvement magnitude
            improvement = result.counterfactual_score - result.actual_score
            priority = 5 - min(4, int(improvement))  # 1-5 scale, lower is higher priority
            
            # Placeholder values
            difficulty = 3  # Medium difficulty
            impact = min(5, int(improvement + 1))  # 1-5 scale based on improvement
            
            # All stages are applicable for now
            applicable_stages = ["planning", "data_preparation", "development", "evaluation", "deployment"]
            
            recommendation = Recommendation(
                id=rec_id,
                title=title,
                description=description,
                source_comparison_id=comparison.id,
                priority=priority,
                difficulty=difficulty,
                impact=impact,
                applicable_stages=applicable_stages
            )
            
            recommendations.append(recommendation)
        
        # Add an overall recommendation if significant improvement
        if comparison.get_weighted_improvement() > 2.0:
            rec_id = f"rec_{comparison.id}_overall"
            
            title = "Implement comprehensive improvements based on counterfactual analysis"
            
            description = (
                f"The counterfactual analysis revealed significant potential for improvement "
                f"with a weighted overall score of {comparison.get_weighted_improvement():.1f}. "
                f"{comparison.overall_assessment}\n\n"
                f"A comprehensive approach addressing multiple dimensions is recommended."
            )
            
            recommendation = Recommendation(
                id=rec_id,
                title=title,
                description=description,
                source_comparison_id=comparison.id,
                priority=1,  # High priority
                difficulty=4,  # Harder due to comprehensive nature
                impact=5,  # High impact
                applicable_stages=["planning", "data_preparation", "development", "evaluation", "deployment"]
            )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def prioritize_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """
        Prioritize recommendations by impact and priority.
        
        Args:
            recommendations: List of recommendations to prioritize
            
        Returns:
            Prioritized list of recommendations
        """
        # Sort by priority-impact ratio (lower is better)
        return sorted(recommendations, key=lambda r: (r.priority_impact_ratio, r.difficulty))
    
    def generate_recommendation_report(self, recommendations: List[Recommendation]) -> str:
        """
        Generate a textual report of recommendations.
        
        Args:
            recommendations: The recommendations to report on
            
        Returns:
            Formatted report text
        """
        if not recommendations:
            return "No recommendations available."
        
        # Generate report
        report = f"# AI/ML Implementation Recommendations\n\n"
        report += f"Based on counterfactual analysis, the following {len(recommendations)} recommendations "
        report += f"are provided to improve implementation outcomes:\n\n"
        
        # Add recommendations in priority order
        prioritized = self.prioritize_recommendations(recommendations)
        
        for i, rec in enumerate(prioritized):
            report += f"## {i+1}. {rec.title}\n\n"
            report += f"{rec.description}\n\n"
            report += f"**Priority:** {'●' * rec.priority}{'○' * (5 - rec.priority)} ({rec.priority}/5)\n"
            report += f"**Impact:** {'★' * rec.impact}{'☆' * (5 - rec.impact)} ({rec.impact}/5)\n"
            report += f"**Difficulty:** {'▲' * rec.difficulty}{'△' * (5 - rec.difficulty)} ({rec.difficulty}/5)\n"
            report += f"**Applicable Stages:** {', '.join(rec.applicable_stages)}\n\n"
            
            if i < len(prioritized) - 1:
                report += "---\n\n"
        
        return report
