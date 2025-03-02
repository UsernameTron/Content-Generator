"""
Domain-specific knowledge evaluators.

These evaluators test the model's knowledge and capabilities in specific domains:
- Customer Experience (CX)
- Artificial Intelligence (AI)
- Machine Learning (ML)
"""

import json
import logging
from pathlib import Path
import random
from evaluators import BaseEvaluator

logger = logging.getLogger("domain_knowledge")

class DomainKnowledgeEvaluator(BaseEvaluator):
    """Base class for domain knowledge evaluators."""
    
    def __init__(self, manager, domain_name):
        """
        Initialize domain knowledge evaluator.
        
        Args:
            manager: The EvaluationManager instance
            domain_name: Name of the domain being evaluated
        """
        super().__init__(manager)
        self.domain_name = domain_name
        self.questions = []
        self.load_questions()
        
    def load_questions(self):
        """Load domain-specific questions from JSON file."""
        try:
            questions_file = Path(f"data/evaluation/{self.domain_name}_questions.json")
            if questions_file.exists():
                with open(questions_file, "r") as f:
                    self.questions = json.load(f)
                logger.info(f"Loaded {len(self.questions)} questions for {self.domain_name}")
            else:
                logger.warning(f"Questions file for {self.domain_name} not found, using default questions")
                # Fall back to default questions defined in the class
                self.questions = self.get_default_questions()
        except Exception as e:
            logger.error(f"Error loading questions for {self.domain_name}: {str(e)}")
            self.questions = self.get_default_questions()
    
    def get_default_questions(self):
        """Get default questions if none are loaded from file."""
        return []
    
    def evaluate(self):
        """
        Evaluate domain knowledge.
        
        Returns:
            dict: Evaluation results
        """
        logger.info(f"Evaluating {self.domain_name} knowledge")
        
        if not self.questions:
            logger.error(f"No questions available for {self.domain_name}")
            return {"score": 0, "error": "No questions available"}
        
        # Sample questions if there are too many
        if len(self.questions) > self.manager.args.batch_size:
            eval_questions = random.sample(self.questions, self.manager.args.batch_size)
        else:
            eval_questions = self.questions
        
        total_score = 0
        question_results = []
        
        for i, question in enumerate(eval_questions):
            logger.info(f"Processing question {i+1}/{len(eval_questions)}: {question['question'][:50]}...")
            
            # Generate response
            response = self.generate_response(question["question"])
            
            # Score the response
            question_score = self.score_response(response, question["criteria"])
            
            # Store results
            question_result = {
                "question": question["question"],
                "response": response,
                "score": question_score,
                "criteria": question["criteria"]
            }
            question_results.append(question_result)
            total_score += question_score
        
        # Calculate average score
        avg_score = total_score / len(eval_questions) if eval_questions else 0
        
        # Prepare detailed metrics
        metrics = {
            "num_questions": len(eval_questions),
            "avg_score": avg_score,
            "max_score": max([q["score"] for q in question_results]) if question_results else 0,
            "min_score": min([q["score"] for q in question_results]) if question_results else 0,
        }
        
        # Prepare summary
        summary = f"Evaluated {len(eval_questions)} questions in {self.domain_name}, average score: {avg_score:.2f}"
        
        return {
            "score": avg_score,
            "metrics": metrics,
            "detailed_results": question_results,
            "summary": summary
        }


class CustomerExperienceEvaluator(DomainKnowledgeEvaluator):
    """Evaluator for Customer Experience domain knowledge."""
    
    def __init__(self, manager):
        """Initialize Customer Experience evaluator."""
        super().__init__(manager, "customer_experience")
    
    def get_default_questions(self):
        """Get default Customer Experience questions."""
        return [
            {
                "question": "What are the key components of a successful Voice of Customer (VoC) program?",
                "criteria": {
                    "data_collection": "collection methods",
                    "analysis": "analysis",
                    "action": "action plans",
                    "feedback_loop": "closed loop",
                    "metrics": "metrics"
                }
            },
            {
                "question": "Explain the difference between Customer Satisfaction (CSAT), Net Promoter Score (NPS), and Customer Effort Score (CES).",
                "criteria": {
                    "csat_def": "satisfaction",
                    "nps_def": "recommend",
                    "ces_def": "effort",
                    "comparison": "different purposes",
                    "use_cases": "when to use"
                }
            },
            {
                "question": "What strategies can a company implement to reduce customer churn?",
                "criteria": {
                    "onboarding": "onboarding",
                    "value": "value demonstration",
                    "feedback": "feedback collection",
                    "proactive": "proactive support",
                    "loyalty": "loyalty program"
                }
            }
        ]


class ArtificialIntelligenceEvaluator(DomainKnowledgeEvaluator):
    """Evaluator for Artificial Intelligence domain knowledge."""
    
    def __init__(self, manager):
        """Initialize Artificial Intelligence evaluator."""
        super().__init__(manager, "artificial_intelligence")
    
    def get_default_questions(self):
        """Get default Artificial Intelligence questions."""
        return [
            {
                "question": "Explain the concept of attention mechanisms in transformer models like GPT and BERT.",
                "criteria": {
                    "self_attention": "self-attention",
                    "multi_head": "multi-head",
                    "parallelization": "parallel",
                    "sequence": "sequence relationships",
                    "scaling": "scaled dot-product"
                }
            },
            {
                "question": "What are the key challenges in implementing Reinforcement Learning from Human Feedback (RLHF)?",
                "criteria": {
                    "reward_modeling": "reward model",
                    "feedback_quality": "quality of feedback",
                    "alignment": "alignment",
                    "scalability": "scalability",
                    "bias": "bias"
                }
            },
            {
                "question": "Compare and contrast supervised learning, unsupervised learning, and reinforcement learning.",
                "criteria": {
                    "supervised_def": "labeled data",
                    "unsupervised_def": "unlabeled data",
                    "reinforcement_def": "reward signals",
                    "applications": "applications",
                    "limitations": "limitations"
                }
            }
        ]


class MachineLearningEvaluator(DomainKnowledgeEvaluator):
    """Evaluator for Machine Learning domain knowledge."""
    
    def __init__(self, manager):
        """Initialize Machine Learning evaluator."""
        super().__init__(manager, "machine_learning")
    
    def get_default_questions(self):
        """Get default Machine Learning questions."""
        return [
            {
                "question": "What is the bias-variance tradeoff in machine learning and how can it be managed?",
                "criteria": {
                    "bias_def": "underfitting",
                    "variance_def": "overfitting",
                    "tradeoff": "balance",
                    "techniques": "cross-validation or regularization",
                    "model_selection": "model complexity"
                }
            },
            {
                "question": "Explain the concept of gradient descent and its variants like SGD, mini-batch, and Adam.",
                "criteria": {
                    "gradient_descent": "optimization algorithm",
                    "sgd": "stochastic",
                    "mini_batch": "mini-batch",
                    "adam": "adaptive",
                    "convergence": "convergence"
                }
            },
            {
                "question": "How do you handle imbalanced datasets in classification problems?",
                "criteria": {
                    "resampling": "oversampling or undersampling",
                    "class_weights": "class weights",
                    "algorithms": "algorithm selection",
                    "evaluation": "appropriate metrics",
                    "data_generation": "synthetic data"
                }
            }
        ]
