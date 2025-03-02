"""
Counterfactual reasoning evaluator.

Tests the model's ability to engage in counterfactual reasoning - considering
alternative scenarios and their implications.
"""

import json
import logging
from pathlib import Path
import random
from evaluators import BaseEvaluator

logger = logging.getLogger("counterfactual")

class CounterfactualEvaluator(BaseEvaluator):
    """Evaluator for counterfactual reasoning capabilities."""
    
    def __init__(self, manager):
        """Initialize counterfactual reasoning evaluator."""
        super().__init__(manager)
        self.scenarios = []
        self.load_scenarios()
    
    def load_scenarios(self):
        """Load counterfactual scenarios from JSON file."""
        try:
            scenarios_file = Path("data/evaluation/counterfactual_scenarios.json")
            if scenarios_file.exists():
                with open(scenarios_file, "r") as f:
                    self.scenarios = json.load(f)
                logger.info(f"Loaded {len(self.scenarios)} counterfactual scenarios")
            else:
                logger.warning("Counterfactual scenarios file not found, using default scenarios")
                self.scenarios = self.get_default_questions()
        except Exception as e:
            logger.error(f"Error loading counterfactual scenarios: {str(e)}")
            self.scenarios = self.get_default_questions()
    
    def get_default_questions(self):
        """Get default counterfactual scenarios.
        
        Returns:
            List of questions/scenarios with criteria
        """
        return [
            {
                "name": "Technology Adoption Timeline",
                "scenario": "Consider a scenario where cloud computing technologies were never developed or adopted. Instead, computing continued to rely primarily on local hardware and traditional data centers.",
                "question": "How would this alternative timeline have affected software development practices, business models, and digital transformation efforts from 2010 to 2025? What technologies might have emerged instead?",
                "criteria": {
                    "software_development": "software development impacts",
                    "business_models": "business model changes",
                    "alternative_tech": "alternative technologies",
                    "historical_context": "historical context",
                    "logical_consistency": "internal consistency"
                }
            },
            {
                "name": "AI Development Path",
                "scenario": "Imagine that early AI research had focused exclusively on symbolic AI and rule-based systems, with no significant investment in neural networks or deep learning approaches.",
                "question": "How would AI capabilities differ today under this counterfactual scenario? What applications would be more advanced, and which would be less developed? How would this have affected recent developments in generative AI?",
                "criteria": {
                    "technical_accuracy": "technically accurate",
                    "comparative_analysis": "compares with actual timeline",
                    "application_differences": "discusses specific applications",
                    "limitations": "addresses limitations",
                    "generative_ai": "discusses generative AI implications"
                }
            },
            {
                "name": "Customer Behavior Shift",
                "scenario": "Consider a world where privacy concerns had become paramount much earlier, with strict global data protection laws equivalent to GDPR being implemented worldwide in 2005 instead of regional regulations emerging gradually.",
                "question": "How would this early privacy-first approach have shaped customer expectations, digital marketing strategies, and the development of personalization technologies? What business models might have become dominant in this scenario?",
                "criteria": {
                    "customer_expectations": "customer behavior changes",
                    "marketing_impact": "marketing strategy changes",
                    "technology_development": "technology development paths",
                    "business_model_innovation": "alternative business models",
                    "causal_reasoning": "clear causal relationships"
                }
            }
        ]
    
    def evaluate(self):
        """
        Evaluate counterfactual reasoning capabilities.
        
        Returns:
            dict: Evaluation results
        """
        logger.info("Evaluating counterfactual reasoning capabilities")
        
        if not self.scenarios:
            logger.error("No counterfactual scenarios available")
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
            prompt = f"""Counterfactual Reasoning Exercise:

{scenario['scenario']}

Question: {scenario['question']}

Please explore this counterfactual scenario thoroughly, considering multiple perspectives and causal relationships. Be specific about how different aspects would have developed differently compared to our actual timeline."""
            
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
        summary = f"Evaluated {len(eval_scenarios)} counterfactual reasoning scenarios, average score: {avg_score:.2f}"
        
        return {
            "score": avg_score,
            "metrics": metrics,
            "detailed_results": scenario_results,
            "summary": summary
        }
