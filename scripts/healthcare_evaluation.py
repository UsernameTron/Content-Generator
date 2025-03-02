#!/usr/bin/env python3
"""
Healthcare-specific evaluation script.
Evaluates model performance on healthcare cross-referencing tasks.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import logging
from rich.logging import RichHandler
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("healthcare-evaluation")

class HealthcareEvaluator:
    """Evaluator for healthcare cross-referencing capabilities."""
    
    def __init__(self, model_path, adapter_path=None, device="mps"):
        """Initialize the healthcare evaluator.
        
        Args:
            model_path: Path to the base model
            adapter_path: Path to the LoRA adapter (if applicable)
            device: Device to run evaluation on (cpu, cuda, mps)
        """
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.device = device
        
        logger.info(f"Initializing healthcare evaluator with model: {model_path}")
        logger.info(f"Device: {device}")
        
        # Set environment variables for Apple Silicon
        if device == "mps":
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map=device
        )
        
        # Load adapter if specified
        if adapter_path:
            logger.info(f"Loading adapter from {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        
        # Create pipeline for generation
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device in ["cuda", "mps"] else -1,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.95,
            top_k=40
        )
        
        logger.info("Model initialized successfully")
    
    def evaluate_contradiction_detection(self, eval_data_path):
        """Evaluate contradiction detection performance.
        
        Args:
            eval_data_path: Path to contradiction evaluation data
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating contradiction detection using {eval_data_path}")
        
        # Load evaluation data
        with open(eval_data_path, 'r') as f:
            eval_data = json.load(f)
        
        # Initialize evaluation metrics
        metrics = {
            "accuracy": 0.0,
            "by_category": {},
            "by_domain": {},
            "confusion_matrix": np.zeros((4, 4)),  # Assuming 4 categories
            "examples": []
        }
        
        # Track unique categories and domains
        categories = set()
        domains = set()
        
        for sample in eval_data:
            categories.add(sample["relationship"])
            domains.add(sample["domain"])
        
        # Initialize category and domain metrics
        for category in categories:
            metrics["by_category"][category] = {"correct": 0, "total": 0, "accuracy": 0.0}
        
        for domain in domains:
            metrics["by_domain"][domain] = {"correct": 0, "total": 0, "accuracy": 0.0}
        
        # Define prompt template
        prompt_template = """Evaluate the relationship between the following two medical statements:

Statement 1: {statement_1}

Statement 2: {statement_2}

Determine if the relationship between these statements is:
A) Supporting - Statement 2 confirms or extends Statement 1
B) Contradicting - Statement 2 disagrees with or negates Statement 1
C) Unrelated - The statements address different topics
D) Temporally Superseded - Statement 2 updates Statement 1 with newer information

Relationship: """
        
        # Category mapping
        category_mapping = {
            "supporting": "A",
            "contradicting": "B",
            "unrelated": "C",
            "temporally_superseded": "D"
        }
        
        reverse_mapping = {v: k for k, v in category_mapping.items()}
        category_indices = {k: i for i, k in enumerate(["supporting", "contradicting", "unrelated", "temporally_superseded"])}
        
        # Evaluate each sample
        correct = 0
        total = 0
        
        for sample in tqdm(eval_data, desc="Evaluating contradiction detection"):
            # Create prompt
            prompt = prompt_template.format(
                statement_1=sample["statement_1"],
                statement_2=sample["statement_2"]
            )
            
            # Get model prediction
            result = self.pipeline(prompt)[0]["generated_text"]
            
            # Extract prediction
            prediction_text = result[len(prompt):].strip()
            
            # Parse prediction (take first letter as answer)
            predicted_letter = prediction_text[0] if prediction_text else ""
            predicted_category = reverse_mapping.get(predicted_letter, "unknown")
            
            # Get true category
            true_category = sample["relationship"]
            
            # Check if correct
            is_correct = predicted_category == true_category
            
            # Update metrics
            if is_correct:
                correct += 1
            total += 1
            
            # Update category metrics
            if true_category in metrics["by_category"]:
                metrics["by_category"][true_category]["total"] += 1
                if is_correct:
                    metrics["by_category"][true_category]["correct"] += 1
            
            # Update domain metrics
            if "domain" in sample and sample["domain"] in metrics["by_domain"]:
                metrics["by_domain"][sample["domain"]]["total"] += 1
                if is_correct:
                    metrics["by_domain"][sample["domain"]]["correct"] += 1
            
            # Update confusion matrix
            if true_category in category_indices and predicted_category in category_indices:
                true_idx = category_indices[true_category]
                pred_idx = category_indices[predicted_category]
                metrics["confusion_matrix"][true_idx, pred_idx] += 1
            
            # Add example
            metrics["examples"].append({
                "statement_1": sample["statement_1"],
                "statement_2": sample["statement_2"],
                "true_category": true_category,
                "predicted_category": predicted_category,
                "correct": is_correct
            })
        
        # Calculate overall accuracy
        metrics["accuracy"] = correct / total if total > 0 else 0.0
        
        # Calculate category accuracies
        for category in metrics["by_category"]:
            cat_metrics = metrics["by_category"][category]
            cat_metrics["accuracy"] = cat_metrics["correct"] / cat_metrics["total"] if cat_metrics["total"] > 0 else 0.0
        
        # Calculate domain accuracies
        for domain in metrics["by_domain"]:
            dom_metrics = metrics["by_domain"][domain]
            dom_metrics["accuracy"] = dom_metrics["correct"] / dom_metrics["total"] if dom_metrics["total"] > 0 else 0.0
        
        logger.info(f"Contradiction detection accuracy: {metrics['accuracy']:.2f}")
        
        return metrics
    
    def evaluate_evidence_ranking(self, eval_data_path):
        """Evaluate evidence strength ranking performance.
        
        Args:
            eval_data_path: Path to evidence ranking evaluation data
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating evidence ranking using {eval_data_path}")
        
        # Load evaluation data
        with open(eval_data_path, 'r') as f:
            eval_data = json.load(f)
        
        # Initialize evaluation metrics
        metrics = {
            "accuracy": 0.0,
            "by_evidence_type": {},
            "by_domain": {},
            "examples": []
        }
        
        # Track unique evidence types and domains
        evidence_types = set()
        domains = set()
        
        for sample in eval_data:
            evidence_types.add(sample["evidence_1"]["type"])
            evidence_types.add(sample["evidence_2"]["type"])
            if "domain" in sample:
                domains.add(sample["domain"])
        
        # Initialize evidence type and domain metrics
        for ev_type in evidence_types:
            metrics["by_evidence_type"][ev_type] = {"correct": 0, "total": 0, "accuracy": 0.0}
        
        for domain in domains:
            metrics["by_domain"][domain] = {"correct": 0, "total": 0, "accuracy": 0.0}
        
        # Define prompt template
        prompt_template = """{task}

Evidence A: {evidence_1}
Evidence B: {evidence_2}

Which evidence (A or B) provides stronger support for clinical decision-making?
Answer with A or B: """
        
        # Evaluate each sample
        correct = 0
        total = 0
        
        for sample in tqdm(eval_data, desc="Evaluating evidence ranking"):
            # Create prompt
            prompt = prompt_template.format(
                task=sample["task"],
                evidence_1=sample["evidence_1"]["description"],
                evidence_2=sample["evidence_2"]["description"]
            )
            
            # Get model prediction
            result = self.pipeline(prompt)[0]["generated_text"]
            
            # Extract prediction
            prediction_text = result[len(prompt):].strip()
            
            # Parse prediction (take first letter as answer)
            predicted_letter = prediction_text[0] if prediction_text else ""
            
            # Determine correct answer
            if sample["evidence_1"]["type"] == sample["stronger_evidence"]:
                correct_letter = "A"
            else:
                correct_letter = "B"
            
            # Check if correct
            is_correct = predicted_letter == correct_letter
            
            # Update metrics
            if is_correct:
                correct += 1
            total += 1
            
            # Update evidence type metrics
            for i, ev_key in enumerate(["evidence_1", "evidence_2"]):
                ev_type = sample[ev_key]["type"]
                if ev_type in metrics["by_evidence_type"]:
                    metrics["by_evidence_type"][ev_type]["total"] += 1
                    if is_correct:
                        metrics["by_evidence_type"][ev_type]["correct"] += 1
            
            # Update domain metrics
            if "domain" in sample and sample["domain"] in metrics["by_domain"]:
                metrics["by_domain"][sample["domain"]]["total"] += 1
                if is_correct:
                    metrics["by_domain"][sample["domain"]]["correct"] += 1
            
            # Add example
            metrics["examples"].append({
                "task": sample["task"],
                "evidence_1": sample["evidence_1"]["description"],
                "evidence_2": sample["evidence_2"]["description"],
                "stronger_evidence": sample["stronger_evidence"],
                "predicted_stronger": "evidence_1" if predicted_letter == "A" else "evidence_2",
                "correct": is_correct
            })
        
        # Calculate overall accuracy
        metrics["accuracy"] = correct / total if total > 0 else 0.0
        
        # Calculate evidence type accuracies
        for ev_type in metrics["by_evidence_type"]:
            type_metrics = metrics["by_evidence_type"][ev_type]
            type_metrics["accuracy"] = type_metrics["correct"] / type_metrics["total"] if type_metrics["total"] > 0 else 0.0
        
        # Calculate domain accuracies
        for domain in metrics["by_domain"]:
            dom_metrics = metrics["by_domain"][domain]
            dom_metrics["accuracy"] = dom_metrics["correct"] / dom_metrics["total"] if dom_metrics["total"] > 0 else 0.0
        
        logger.info(f"Evidence ranking accuracy: {metrics['accuracy']:.2f}")
        
        return metrics
    
    def run_all_evaluations(self, data_dir, output_dir):
        """Run all healthcare evaluations.
        
        Args:
            data_dir: Directory containing evaluation data
            output_dir: Directory to save evaluation results
            
        Returns:
            results: Dictionary of all evaluation results
        """
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Running all healthcare evaluations with data from {data_dir}")
        
        # Initialize results
        results = {
            "contradiction_detection": None,
            "evidence_ranking": None,
            "metadata": {
                "model_path": self.model_path,
                "adapter_path": self.adapter_path,
                "device": self.device,
                "evaluation_timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        # Evaluate contradiction detection
        contradiction_path = data_dir / "contradiction_eval.json"
        if contradiction_path.exists():
            results["contradiction_detection"] = self.evaluate_contradiction_detection(contradiction_path)
        else:
            logger.warning(f"Contradiction evaluation data not found: {contradiction_path}")
        
        # Evaluate evidence ranking
        evidence_path = data_dir / "evidence_ranking_eval.json"
        if evidence_path.exists():
            results["evidence_ranking"] = self.evaluate_evidence_ranking(evidence_path)
        else:
            logger.warning(f"Evidence ranking evaluation data not found: {evidence_path}")
        
        # Save results
        timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
        results_path = output_dir / f"healthcare_eval_{timestamp}.json"
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {results_path}")
        
        return results

def main():
    """Main function to run the healthcare evaluation."""
    parser = argparse.ArgumentParser(description="Healthcare-specific model evaluation")
    parser.add_argument("--model", type=str, required=True,
                       help="Path or name of the base model")
    parser.add_argument("--adapter", type=str, default=None,
                       help="Path to LoRA adapter (if applicable)")
    parser.add_argument("--data_dir", type=str, default="data/healthcare",
                       help="Directory containing evaluation data")
    parser.add_argument("--output_dir", type=str, default="evaluation_results/healthcare",
                       help="Directory to save evaluation results")
    parser.add_argument("--device", type=str, default="mps",
                       help="Device to run evaluation on (cpu, cuda, mps)")
    
    args = parser.parse_args()
    
    try:
        evaluator = HealthcareEvaluator(args.model, args.adapter, args.device)
        results = evaluator.run_all_evaluations(args.data_dir, args.output_dir)
        print(f"Evaluation complete with overall results:")
        
        if results["contradiction_detection"]:
            print(f"Contradiction Detection: {results['contradiction_detection']['accuracy']:.2f}")
        
        if results["evidence_ranking"]:
            print(f"Evidence Ranking: {results['evidence_ranking']['accuracy']:.2f}")
        
    except Exception as e:
        logger.error(f"Error during healthcare evaluation: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
