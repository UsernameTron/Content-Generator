"""
Base Evaluator class for the comprehensive evaluation framework.
This abstract class provides the foundation for all specific evaluators.
"""

import abc
import json
import logging
import os
import torch
from typing import Dict, List, Any, Tuple, Optional

from rich.logging import RichHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)

class BaseEvaluator(abc.ABC):
    """Abstract base class for all evaluators."""
    
    def __init__(self, manager, evaluator_name=None):
        """
        Initialize the base evaluator.
        
        Args:
            manager: The evaluation manager instance
            evaluator_name: Optional name of the evaluator (defaults to class name)
        """
        # Set up the manager reference
        self.manager = manager
        
        # Set up the evaluator name
        if evaluator_name is None:
            evaluator_name = self.__class__.__name__.replace('Evaluator', '').lower()
        self.evaluator_name = evaluator_name
        self.logger = logging.getLogger(f"evaluator.{evaluator_name}")
        
        # Get model, tokenizer, and device from manager if available
        self.model = getattr(manager, 'model', None)
        self.tokenizer = getattr(manager, 'tokenizer', None)
        self.device = getattr(manager, 'device', None)
        
        # Initialize results dictionary
        self.results = {}
        
    def set_model(self, model, tokenizer=None, device=None):
        """
        Set or update the model, tokenizer, and device.
        
        Args:
            model: The model to evaluate
            tokenizer: The tokenizer for the model
            device: The device to run evaluation on
        """
        self.model = model
        if tokenizer:
            self.tokenizer = tokenizer
        if device:
            self.device = device
            
    @abc.abstractmethod
    def evaluate(self, **kwargs) -> Dict[str, Any]:
        """
        Evaluate the model and return results.
        
        Returns:
            Dict containing evaluation results
        """
        pass
    
    @abc.abstractmethod
    def get_default_questions(self) -> List[Dict[str, Any]]:
        """
        Get the default questions/scenarios for this evaluator.
        
        Returns:
            List of questions/scenarios with criteria
        """
        pass
    
    def load_questions_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load questions/scenarios from a JSON file.
        
        Args:
            file_path: Path to the JSON file containing questions
            
        Returns:
            List of questions/scenarios
        """
        if not os.path.exists(file_path):
            self.logger.warning(f"File {file_path} not found. Using default questions.")
            return self.get_default_questions()
        
        try:
            with open(file_path, 'r') as f:
                questions = json.load(f)
            self.logger.info(f"Loaded {len(questions)} questions from {file_path}")
            return questions
        except Exception as e:
            self.logger.error(f"Error loading questions from {file_path}: {e}")
            return self.get_default_questions()
    
    def get_questions(self, file_path: Optional[str] = None, 
                     batch_size: int = 5) -> List[Dict[str, Any]]:
        """
        Get questions either from file or defaults, limited by batch size.
        
        Args:
            file_path: Optional path to questions file
            batch_size: Number of questions to use
            
        Returns:
            List of questions limited by batch_size
        """
        if file_path and os.path.exists(file_path):
            questions = self.load_questions_from_file(file_path)
        else:
            questions = self.get_default_questions()
            
        # Limit to batch_size (but ensure at least 1)
        batch_size = max(1, batch_size)
        questions = questions[:batch_size]
        
        self.logger.info(f"Using {len(questions)} questions for evaluation")
        return questions
    
    def generate_response(self, prompt: str, 
                          temperature: float = 0.7, 
                          max_tokens: int = 1024) -> str:
        """
        Generate a response from the model.
        
        Args:
            prompt: The input prompt
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            The generated response
        """
        # Use the manager's generate_response method if available
        if hasattr(self.manager, 'generate_response'):
            return self.manager.generate_response(prompt, max_tokens=max_tokens)
        
        # Fallback to direct generation if manager's method is not available
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer must be set before generating responses")
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate with specified parameters
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=(temperature > 0),
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the output
            response = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], 
                                            skip_special_tokens=True)
            return response.strip()
        
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def score_response(self, response: str, criteria: Dict[str, str]) -> Tuple[float, Dict[str, float]]:
        """
        Score a response based on the given criteria.
        Each criterion is checked for presence in the response.
        
        Args:
            response: The model's response
            criteria: Dictionary of criteria and keywords to check for
            
        Returns:
            Tuple of (overall_score, criteria_scores)
        """
        criteria_scores = {}
        response_lower = response.lower()
        
        # Check each criterion
        for key, keywords in criteria.items():
            # Convert single keyword to list
            if isinstance(keywords, str):
                keywords = [keywords]
                
            # Check if any of the keywords are present
            score = 0.0
            for keyword in keywords:
                if keyword.lower() in response_lower:
                    score = 1.0
                    break
                    
            criteria_scores[key] = score
            
        # Calculate overall score as average of criteria scores
        if criteria_scores:
            overall_score = sum(criteria_scores.values()) / len(criteria_scores)
        else:
            overall_score = 0.0
            
        return overall_score, criteria_scores
        
    def format_results(self) -> Dict[str, Any]:
        """
        Format results for reporting.
        
        Returns:
            Dictionary with formatted results
        """
        if not self.results:
            return {"evaluator": self.evaluator_name, "status": "No results"}
            
        return {
            "evaluator": self.evaluator_name,
            "results": self.results
        }
