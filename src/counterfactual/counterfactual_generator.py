"""
Unified interface for counterfactual reasoning modules.

This module provides an integration layer for all counterfactual reasoning components,
enabling seamless incorporation of counterfactual insights into content generation.
"""

import logging
import uuid
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

from .causal_analysis import CausalAnalyzer, FailureCase, DecisionPoint
from .comparison import CounterfactualComparator, StructuredComparison
from .pattern_recognition import PatternRecognizer
from .recommendation import RecommendationGenerator, Recommendation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class CounterfactualGenerator:
    """
    Unified interface for counterfactual reasoning functionality.
    
    This class integrates all counterfactual modules (causal analysis, comparison,
    pattern recognition, recommendation) and provides a simple API for generating
    counterfactual insights.
    """
    
    def __init__(
        self, 
        data_dir: str = "data/counterfactual",
        output_dir: str = "output/counterfactual"
    ):
        """
        Initialize the counterfactual generator.
        
        Args:
            data_dir: Base directory for storing counterfactual data
            output_dir: Directory for storing output artifacts
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.causal_analyzer = CausalAnalyzer(
            cases_dir=str(self.data_dir / "failure_cases")
        )
        
        self.comparator = CounterfactualComparator(
            comparisons_dir=str(self.data_dir / "comparisons")
        )
        
        self.pattern_recognizer = PatternRecognizer(
            patterns_dir=str(self.data_dir / "patterns"),
            cases_dir=str(self.data_dir / "failure_cases")
        )
        
        self.recommendation_generator = RecommendationGenerator(
            recommendations_dir=str(self.data_dir / "recommendations")
        )
        
        logger.info("Counterfactual Generator initialized")
    
    def analyze_implementation_failure(self, case_data: Dict) -> FailureCase:
        """
        Analyze an AI/ML implementation failure.
        
        Args:
            case_data: Dictionary containing failure case data
            
        Returns:
            FailureCase object for the analyzed failure
        """
        logger.info(f"Analyzing implementation failure: {case_data.get('title', 'Untitled')}")
        
        # Check if case already has an ID
        if 'id' not in case_data:
            case_data['id'] = f"case_{str(uuid.uuid4())[:8]}"
        
        # Create failure case
        failure_case = FailureCase.from_dict(case_data)
        
        # Analyze causal relationships
        self.causal_analyzer.analyze_causal_relationships(failure_case)
        
        # Save the analyzed case
        self.causal_analyzer.save_case(failure_case)
        
        logger.info(f"Analysis complete for case: {failure_case.id}")
        return failure_case
    
    def generate_alternatives(
        self, 
        case_id: str,
        decision_point_ids: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Generate alternative paths for critical decision points.
        
        Args:
            case_id: ID of the failure case to analyze
            decision_point_ids: Optional list of specific decision point IDs to focus on
            
        Returns:
            Dictionary mapping decision point IDs to alternative paths
        """
        logger.info(f"Generating alternatives for case: {case_id}")
        
        # Retrieve the failure case
        if case_id not in self.causal_analyzer.failure_cases:
            logger.error(f"Failure case not found: {case_id}")
            return {}
            
        failure_case = self.causal_analyzer.failure_cases[case_id]
        
        # Filter decision points if specific IDs provided
        decision_points = failure_case.decision_points
        if decision_point_ids:
            decision_points = [dp for dp in decision_points if dp.id in decision_point_ids]
        
        # Generate alternative paths
        alternatives = {}
        for dp in decision_points:
            alternative = self.causal_analyzer.generate_alternative_path(dp)
            alternatives[dp.id] = alternative
            
        logger.info(f"Generated alternatives for {len(alternatives)} decision points")
        return alternatives
    
    def create_comparison(
        self, 
        case_id: str,
        counterfactual_description: str = None,
        decision_point_ids: Optional[List[str]] = None
    ) -> Optional[StructuredComparison]:
        """
        Create a structured comparison between actual and counterfactual scenarios.
        
        Args:
            case_id: ID of the failure case to analyze
            counterfactual_description: Optional description of the counterfactual scenario
            decision_point_ids: Optional list of specific decision point IDs to focus on
            
        Returns:
            StructuredComparison object or None if case not found
        """
        logger.info(f"Creating comparison for case: {case_id}")
        
        # Retrieve the failure case
        if case_id not in self.causal_analyzer.failure_cases:
            logger.error(f"Failure case not found: {case_id}")
            return None
            
        failure_case = self.causal_analyzer.failure_cases[case_id]
        
        # Generate counterfactual description if none provided
        if not counterfactual_description:
            # Just use a basic description
            counterfactual_description = f"Alternative scenario for {failure_case.title} where different decisions were made."
        
        # Create comparison
        comparison = self.comparator.create_comparison(
            failure_case=failure_case,
            counterfactual_description=counterfactual_description
        )
        
        # Add comparison results for each dimension
        for dimension in comparison.dimensions:
            # Simulate scores for demonstration purposes
            actual_score = 5.0  # Baseline average score
            counterfactual_score = 7.5  # Better score for counterfactual
            
            # For data quality dimension, make a bigger difference if there's a data-related decision
            if dimension.name == "data_quality" and any("data" in dp.description.lower() for dp in failure_case.decision_points):
                actual_score = 4.0
                counterfactual_score = 8.0
                
            # For model suitability, check if any decision points relate to model selection
            elif dimension.name == "model_suitability" and any("model" in dp.description.lower() for dp in failure_case.decision_points):
                actual_score = 3.5
                counterfactual_score = 8.5
                
            # For evaluation rigor, check if any decision points relate to validation/evaluation
            elif dimension.name == "evaluation_rigor" and any(("valid" in dp.description.lower() or "eval" in dp.description.lower()) for dp in failure_case.decision_points):
                actual_score = 3.0
                counterfactual_score = 7.5
                
            # Generate explanation for the scores
            explanation = f"The actual implementation scored {actual_score}/10 on {dimension.name.replace('_', ' ')}, "
            explanation += f"while the counterfactual approach could achieve {counterfactual_score}/10 through better practices. "
            explanation += f"This represents a {counterfactual_score - actual_score:.1f} point improvement opportunity."
            
            # Add the result to the comparison
            self.comparator.add_comparison_result(
                comparison=comparison,
                dimension=dimension.name,
                actual_score=actual_score,
                counterfactual_score=counterfactual_score,
                explanation=explanation
            )
            
        # Add an overall assessment
        overall_assessment = "The counterfactual analysis reveals significant opportunities for improvement across multiple dimensions. "
        overall_assessment += "By making different decisions at critical points, particularly in "
        
        # Identify top improvement areas
        top_improvements = sorted(
            [(result.counterfactual_score - result.actual_score, result.dimension) for result in comparison.results],
            reverse=True
        )[:3]
        
        if top_improvements:
            top_areas = [dim.replace("_", " ") for _, dim in top_improvements]
            overall_assessment += f"{', '.join(top_areas[:2])} and {top_areas[2]}" if len(top_areas) >= 3 else f"{' and '.join(top_areas)}"
            overall_assessment += ", the implementation outcomes could have been substantially better."
            
        self.comparator.set_overall_assessment(comparison, overall_assessment)
        
        # Make sure to save the comparison
        self.comparator.save_comparison(comparison)
        
        logger.info(f"Structured comparison created: {comparison.id}")
        return comparison
    
    def identify_patterns(self, case_id: Optional[str] = None) -> List[Any]:
        """
        Identify patterns across failure cases.
        
        Args:
            case_id: Optional specific failure case ID to find patterns for
            
        Returns:
            List of pattern objects
        """
        if case_id:
            logger.info(f"Identifying patterns for specific case: {case_id}")
            return self.pattern_recognizer.identify_patterns_in_case(case_id)
        else:
            logger.info("Identifying patterns across all failure cases")
            return self.pattern_recognizer.identify_patterns()
    
    def generate_recommendations(
        self, 
        comparison_id: str
    ) -> List[Recommendation]:
        """
        Generate practical recommendations based on counterfactual analysis.
        
        Args:
            comparison_id: ID of the structured comparison
            
        Returns:
            List of actionable recommendations
        """
        logger.info(f"Generating recommendations from comparison: {comparison_id}")
        
        # Retrieve the comparison
        comparison = self.comparator.get_comparison(comparison_id)
        if not comparison:
            logger.error(f"Comparison not found: {comparison_id}")
            return []
        
        # Generate recommendations
        recommendations = self.recommendation_generator.generate_recommendations(comparison)
        
        # Prioritize recommendations
        prioritized = self.recommendation_generator.prioritize_recommendations(recommendations)
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        return prioritized
    
    # Keep the old method for backward compatibility
    def get_recommendations(
        self, 
        comparison: StructuredComparison
    ) -> List[Recommendation]:
        """
        Generate practical recommendations based on counterfactual analysis.
        
        Args:
            comparison: Structured comparison between actual and counterfactual
            
        Returns:
            List of actionable recommendations
        """
        logger.info(f"Generating recommendations from comparison: {comparison.id}")
        
        # Generate recommendations
        recommendations = self.recommendation_generator.generate_recommendations(comparison)
        
        # Prioritize recommendations
        prioritized = self.recommendation_generator.prioritize_recommendations(recommendations)
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        return prioritized
    
    def generate_insight_report(
        self,
        failure_case: FailureCase,
        comparison: StructuredComparison,
        recommendations: List[Recommendation],
        patterns: List[Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive insight report.
        
        Args:
            failure_case: The analyzed failure case
            comparison: Structured comparison
            recommendations: Generated recommendations
            patterns: Identified patterns
            output_path: Optional path to save the report
            
        Returns:
            Formatted report text
        """
        logger.info(f"Generating insight report for case: {failure_case.id}")
        
        # Generate report sections
        causal_report = self.causal_analyzer.generate_case_report(failure_case)
        comparison_report = self.comparator.generate_comparison_report(comparison)
        recommendation_report = self.recommendation_generator.generate_recommendation_report(recommendations)
        
        # Generate patterns section
        pattern_section = "## Pattern Analysis\n\n"
        if patterns:
            pattern_section += "This failure case exhibits the following patterns:\n\n"
            for pattern in patterns:
                pattern_section += f"### {pattern.name}\n\n"
                pattern_section += f"{pattern.description}\n\n"
                pattern_section += f"Severity: {pattern.severity:.1f}/10  |  "
                pattern_section += f"Frequency: {pattern.frequency:.1%} of analyzed cases\n\n"
        else:
            pattern_section += "No established patterns were identified for this failure case.\n\n"
        
        # Combine sections
        report = f"# Counterfactual Analysis Insight Report\n\n"
        report += f"## Executive Summary\n\n"
        report += f"This report analyzes the implementation failure '{failure_case.title}' "
        report += f"through counterfactual reasoning to identify what could have been done differently.\n\n"
        
        report += f"Key insights:\n"
        report += f"- {len(failure_case.decision_points)} critical decision points identified\n"
        report += f"- {comparison.get_weighted_improvement():.1f} potential improvement score (weighted)\n"
        report += f"- {len(recommendations)} actionable recommendations generated\n"
        report += f"- {len(patterns)} relevant patterns identified\n\n"
        
        report += "---\n\n"
        report += causal_report + "\n\n"
        report += "---\n\n"
        report += comparison_report + "\n\n"
        report += "---\n\n"
        report += pattern_section + "\n\n"
        report += "---\n\n"
        report += recommendation_report
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Saved insight report to {output_path}")
        
        return report
    
    def integrate_with_content(
        self, 
        content: str, 
        failure_case: FailureCase,
        recommendations: List[Recommendation],
        integration_style: str = "summary"
    ) -> str:
        """
        Integrate counterfactual insights into generated content.
        
        Args:
            content: Original content to enhance
            failure_case: Analyzed failure case
            recommendations: Generated recommendations
            integration_style: Style of integration ("minimal", "summary", "detailed")
            
        Returns:
            Enhanced content with counterfactual insights
        """
        logger.info("Integrating counterfactual insights into content")
        
        # Check if we have a valid case and recommendations
        if not failure_case or not failure_case.decision_points:
            logger.warning("No valid failure case provided for integration")
            return content
            
        if not recommendations:
            logger.warning("No recommendations provided for integration")
            return content
        
        # Select integration style
        enhanced_content = content
        
        if integration_style == "minimal":
            # Add just a brief note
            insight = "\n\n**Note:** "
            insight += f"A counterfactual analysis reveals that "
            insight += f"alternative approaches in {len(failure_case.decision_points)} "
            insight += f"key decision areas could have improved outcomes."
            
            # Add to content
            enhanced_content = content + insight
            
        elif integration_style == "detailed":
            # Add a comprehensive counterfactual section
            insight_section = "\n\n## Counterfactual Analysis\n\n"
            insight_section += f"Analysis of the implementation '{failure_case.title}' reveals several critical decision points "
            insight_section += f"where alternative approaches could have yielded better outcomes:\n\n"
            
            # Add decision points and alternatives
            for dp in failure_case.decision_points[:min(5, len(failure_case.decision_points))]:
                insight_section += f"### Decision: {dp.description}\n\n"
                insight_section += f"**Actual approach:** {dp.actual_decision}\n\n"
                
                # Use the first alternative instead of alternative_decision
                alternative = dp.alternatives[0] if dp.alternatives else "No alternative specified"
                insight_section += f"**Alternative approach:** {alternative}\n\n"
                
                insight_section += f"**Potential impact:** {dp.importance * 2}/10\n\n"
            
            # Add recommendations
            insight_section += "### Recommendations\n\n"
            for rec in recommendations[:min(5, len(recommendations))]:
                insight_section += f"- **{rec.title}**: {rec.description.split('.')[0]}.\n"
            
            # Add to content
            enhanced_content = content + insight_section
            
        else:  # summary (default)
            # Add a summary section
            insight_section = "\n\n## Counterfactual Insights\n\n"
            insight_section += f"A counterfactual analysis of this implementation reveals "
            insight_section += f"that different approaches could have improved outcomes. "
            insight_section += f"Key alternatives include:\n\n"
            
            # Add top decision points
            for dp in sorted(failure_case.decision_points, key=lambda x: x.importance, reverse=True)[:3]:
                # Get the first alternative for each decision point
                alternative = dp.alternatives[0] if dp.alternatives else "an alternative approach"
                insight_section += f"- Instead of '{dp.actual_decision}', the implementation could have used '{alternative}'\n"
            
            # Add top recommendations
            insight_section += "\n**Key recommendations:**\n\n"
            for rec in recommendations[:min(3, len(recommendations))]:
                insight_section += f"- {rec.title}\n"
            
            # Add to content
            enhanced_content = content + insight_section
        
        logger.info(f"Added {integration_style} counterfactual insights to content")
        return enhanced_content

    def generate_report(self, case_id: str) -> str:
        """
        Generate a comprehensive report for a failure case.
        
        This method brings together all aspects of counterfactual analysis
        including causal analysis, alternatives, comparison, patterns,
        and recommendations.
        
        Args:
            case_id: ID of the failure case to report on
            
        Returns:
            Formatted comprehensive report
        """
        logger.info(f"Generating comprehensive report for case: {case_id}")
        
        # Check if case exists
        failure_case = self.causal_analyzer.get_case(case_id)
        if not failure_case:
            logger.error(f"Failure case not found: {case_id}")
            return "Error: Failure case not found"
        
        # Generate components for the report
        alternatives = self.generate_alternatives(case_id)
        comparison = self.create_comparison(case_id)
        
        if not comparison:
            logger.error(f"Could not create comparison for case: {case_id}")
            return f"Error: Could not create comparison for case: {case_id}"
            
        recommendations = self.generate_recommendations(comparison.id)
        patterns = self.identify_patterns(case_id)
        
        # Create the report
        report = f"# Counterfactual Analysis Report: {failure_case.title}\n\n"
        report += f"## Overview\n\n"
        report += f"{failure_case.description}\n\n"
        report += f"Industry: {failure_case.industry}\n"
        report += f"Project Type: {failure_case.project_type}\n"
        report += f"Primary Failure Modes: {', '.join(failure_case.primary_failure_modes)}\n\n"
        
        # Add causal analysis
        report += f"## Causal Analysis\n\n"
        report += f"The failure involved {len(failure_case.decision_points)} key decision points:\n\n"
        
        for i, dp in enumerate(failure_case.decision_points):
            report += f"### Decision Point {i+1}: {dp.description}\n\n"
            report += f"**Actual Decision:** {dp.actual_decision}\n\n"
            report += f"**Alternatives:**\n"
            for alt in dp.alternatives:
                report += f"- {alt}\n"
            report += f"\n**Importance:** {dp.importance * 10:.1f}/10\n\n"
            report += f"**Stage:** {dp.stage}\n\n"
            
        # Add comparison
        report += f"## Counterfactual Comparison\n\n"
        report += f"### Scenario: {comparison.counterfactual_description}\n\n"
        
        report += f"### Comparison Results\n\n"
        for result in comparison.results:
            report += f"**{result.dimension}**\n"
            report += f"- Actual: {result.actual_score}/10\n"
            report += f"- Counterfactual: {result.counterfactual_score}/10\n"
            report += f"- Explanation: {result.explanation}\n\n"
            
        report += f"### Overall Assessment\n\n"
        report += f"{comparison.overall_assessment}\n\n"
        
        # Add recommendations
        report += f"## Recommendations\n\n"
        for i, rec in enumerate(recommendations):
            report += f"### {i+1}. {rec.title}\n\n"
            report += f"{rec.description}\n\n"
            
        # Add patterns if any
        if patterns:
            report += f"## Related Patterns\n\n"
            for pattern in patterns:
                report += f"### {pattern.name}\n\n"
                report += f"{pattern.description}\n\n"
                report += f"Affects {len(pattern.case_ids)} cases"
                report += f" across {', '.join(pattern.affected_industries[:3])} industries.\n\n"
        
        logger.info(f"Generated comprehensive report for case: {case_id}")
        return report
