"""
Cross-referencing capability evaluator.

Tests the model's ability to perform cross-referencing across different sources
and domains of information.
"""

import json
import logging
from pathlib import Path
import random
from evaluators import BaseEvaluator

logger = logging.getLogger("cross_referencing")

class CrossReferencingEvaluator(BaseEvaluator):
    """Evaluator for cross-referencing capabilities."""
    
    def __init__(self, manager):
        """Initialize cross-referencing evaluator."""
        super().__init__(manager)
        self.scenarios = []
        self.load_scenarios()
    
    def load_scenarios(self):
        """Load cross-referencing scenarios from JSON file."""
        try:
            scenarios_file = Path("data/evaluation/cross_reference_scenarios.json")
            if scenarios_file.exists():
                with open(scenarios_file, "r") as f:
                    self.scenarios = json.load(f)
                logger.info(f"Loaded {len(self.scenarios)} cross-referencing scenarios")
            else:
                logger.warning("Cross-referencing scenarios file not found, using default scenarios")
                self.scenarios = self.get_default_questions()
        except Exception as e:
            logger.error(f"Error loading cross-referencing scenarios: {str(e)}")
            self.scenarios = self.get_default_questions()
    
    def get_default_questions(self):
        """Get default cross-referencing scenarios.
        
        Returns:
            List of questions/scenarios with criteria
        """
        return [
            {
                "name": "Industry Trend Analysis",
                "context": [
                    "Report A indicates a 15% increase in cloud computing adoption across healthcare in 2024.",
                    "Report B shows that 65% of healthcare providers have concerns about data security in cloud environments.",
                    "Report C states that regulatory compliance costs for healthcare IT have increased by 23% since 2023."
                ],
                "question": "Based on these three reports, what strategy should healthcare providers consider for their cloud computing initiatives, and what are the key considerations they should prioritize?",
                "criteria": {
                    "integration": "connects all three reports",
                    "security": "addresses security concerns",
                    "compliance": "mentions regulatory compliance",
                    "cost_benefit": "discusses cost implications",
                    "recommendation": "clear recommendation"
                }
            },
            {
                "name": "Contradictory Research",
                "context": [
                    "Study A concludes that remote work increases productivity by 22% for software development teams.",
                    "Study B finds a 12% decrease in code quality and 18% more bugs in software developed by remote teams.",
                    "Study C shows no statistical difference in productivity between remote and in-office teams, but identifies communication challenges in remote settings."
                ],
                "question": "Given these seemingly contradictory research findings, what can we conclude about remote work for software development teams? How should an organization approach remote work policies?",
                "criteria": {
                    "contradiction_recognition": "acknowledges contradictions",
                    "methodology": "considers possible methodology differences",
                    "nuanced_view": "presents nuanced perspective",
                    "contextual_factors": "discusses contextual factors",
                    "practical_recommendation": "provides practical advice"
                }
            },
            {
                "name": "Multi-domain Synthesis",
                "context": [
                    "AI research indicates that large language models demonstrate emergent reasoning capabilities at scale.",
                    "Cognitive science research suggests human reasoning relies heavily on analogical thinking and pattern matching.",
                    "Philosophy of mind debates the difference between simulation of understanding and actual understanding."
                ],
                "question": "By synthesizing insights from AI research, cognitive science, and philosophy, evaluate whether current AI systems can be said to 'understand' in any meaningful sense. What implications does this have for AI development?",
                "criteria": {
                    "interdisciplinary": "connects all three domains",
                    "technical_accuracy": "accurately represents AI capabilities",
                    "cognitive_science": "applies cognitive science concepts",
                    "philosophical_depth": "engages with philosophical questions",
                    "implications": "discusses practical implications"
                }
            }
        ]
    
    def evaluate(self):
        """
        Evaluate cross-referencing capabilities.
        
        Returns:
            dict: Evaluation results
        """
        logger.info("Evaluating cross-referencing capabilities")
        
        if not self.scenarios:
            logger.error("No cross-referencing scenarios available")
            return {"score": 0, "error": "No scenarios available"}
        
        # Sample scenarios if there are too many
        if len(self.scenarios) > self.manager.args.batch_size:
            eval_scenarios = random.sample(self.scenarios, self.manager.args.batch_size)
        else:
            eval_scenarios = self.scenarios
        
        total_score = 0
        scenario_results = []
        
        for i, scenario in enumerate(eval_scenarios):
            logger.info(f"Processing scenario {i+1}/{len(eval_scenarios)}: {scenario['name']}")
            
            # Prepare prompt
            context_text = "\n\n".join([f"Source {idx+1}: {ctx}" for idx, ctx in enumerate(scenario["context"])])
            prompt = f"""Please analyze the following information from multiple sources:

{context_text}

Question: {scenario['question']}

Please be thorough in your analysis and make sure to connect insights from all sources."""
            
            # Generate response
            response = self.generate_response(prompt, max_tokens=1024)
            
            # Score the response
            scenario_score = self.score_response(response, scenario["criteria"])
            
            # Store results
            scenario_result = {
                "name": scenario["name"],
                "prompt": prompt,
                "response": response,
                "score": scenario_score,
                "criteria": scenario["criteria"]
            }
            scenario_results.append(scenario_result)
            total_score += scenario_score
        
        # Calculate average score
        avg_score = total_score / len(eval_scenarios) if eval_scenarios else 0
        
        # Prepare detailed metrics
        metrics = {
            "num_scenarios": len(eval_scenarios),
            "avg_score": avg_score,
            "max_score": max([s["score"] for s in scenario_results]) if scenario_results else 0,
            "min_score": min([s["score"] for s in scenario_results]) if scenario_results else 0,
        }
        
        # Prepare summary
        summary = f"Evaluated {len(eval_scenarios)} cross-referencing scenarios, average score: {avg_score:.2f}"
        
        return {
            "score": avg_score,
            "metrics": metrics,
            "detailed_results": scenario_results,
            "summary": summary
        }
