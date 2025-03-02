#!/usr/bin/env python3
"""
Demonstrate healthcare continuous learning capabilities.
This script shows how the continuous learning system works by:
1. Generating synthetic evaluation data
2. Analyzing model performance
3. Creating new training examples for weak areas
4. Updating the training dataset
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table
from visualize_metrics import HealthcareContinuousLearning
from generate_synthetic_evaluation import generate_synthetic_evaluation
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("healthcare-learning-demo")
console = Console()

def analyze_performance(evaluation_results, improvement_threshold=0.10):
    """Simplified analysis of performance to identify improvement areas.
    
    Args:
        evaluation_results: Results from model evaluation
        improvement_threshold: Threshold for identifying areas needing improvement
        
    Returns:
        Dictionary with analysis results
    """
    target_accuracy = 0.90  # Target accuracy threshold
    
    # Extract category and domain metrics
    category_metrics = evaluation_results.get("by_category", {})
    domain_metrics = evaluation_results.get("by_domain", {})
    
    # Find categories and domains below threshold
    improvement_areas = []
    
    # Check categories
    for category, metrics in category_metrics.items():
        accuracy = metrics.get("accuracy", 0)
        if accuracy < target_accuracy - improvement_threshold:
            improvement_areas.append({
                "type": "category",
                "name": category,
                "current_accuracy": accuracy,
                "target_accuracy": target_accuracy
            })
    
    # Check domains
    for domain, metrics in domain_metrics.items():
        accuracy = metrics.get("accuracy", 0)
        if accuracy < target_accuracy - improvement_threshold:
            improvement_areas.append({
                "type": "domain",
                "name": domain,
                "current_accuracy": accuracy,
                "target_accuracy": target_accuracy
            })
    
    # Prioritize improvement areas
    improvement_areas.sort(key=lambda x: x["current_accuracy"])
    
    return {
        "overall_accuracy": evaluation_results.get("accuracy", 0),
        "improvement_areas": improvement_areas,
        "examples": evaluation_results.get("examples", [])
    }

def generate_examples(analysis, count=20):
    """Generate new examples based on analysis results.
    
    Args:
        analysis: Analysis results from analyze_performance
        count: Number of examples to generate
        
    Returns:
        List of new examples
    """
    # Extract data for generation
    improvement_areas = analysis.get("improvement_areas", [])
    examples = analysis.get("examples", [])
    
    # Determine focus areas
    focus_categories = [area["name"] for area in improvement_areas if area["type"] == "category"]
    focus_domains = [area["name"] for area in improvement_areas if area["type"] == "domain"]
    
    # Default categories and domains if none identified
    categories = ["supporting", "contradicting", "unrelated", "temporally_superseded"] 
    domains = ["cardiology", "oncology", "neurology", "infectious_disease", "pharmacology"]
    
    if not focus_categories:
        focus_categories = categories
    if not focus_domains:
        focus_domains = domains
    
    # Choose templates from existing examples
    templates = []
    for example in examples:
        true_category = example.get("true_category", "")
        domain = example.get("domain", "")
        
        # Prioritize examples from weak areas
        if true_category in focus_categories or domain in focus_domains:
            templates.append(example)
    
    # If no suitable templates, use all examples
    if not templates and examples:
        templates = examples
    
    # Generate new examples
    new_examples = []
    for _ in range(count):
        if not templates:
            break
            
        # Select a random template
        template = random.choice(templates)
        
        # Generate a variation
        variation = create_example_variation(template, focus_categories, focus_domains)
        if variation:
            new_examples.append(variation)
    
    return new_examples

def create_example_variation(template, focus_categories, focus_domains):
    """Create a variation of an example template.
    
    Args:
        template: Template example
        focus_categories: Categories to focus on
        focus_domains: Domains to focus on
        
    Returns:
        New example variation
    """
    # Extract information from template
    template_category = template.get("true_category", "")
    template_domain = template.get("domain", "")
    
    # Determine category and domain for new example
    if focus_categories and random.random() < 0.7:  # 70% chance to use focus category
        category = random.choice(focus_categories)
    else:
        category = template_category
    
    if focus_domains and random.random() < 0.7:  # 70% chance to use focus domain
        domain = random.choice(focus_domains)
    else:
        domain = template_domain
    
    # Create variations of the statements
    stmt1 = template.get("statement_1", "")
    stmt2 = template.get("statement_2", "")
    
    # Modify statements slightly
    statement1 = modify_statement(stmt1)
    statement2 = modify_statement(stmt2)
    
    # Create the new example
    example = {
        "statement1": statement1,
        "statement2": statement2,
        "type": category,
        "domain": domain
    }
    
    return example

def modify_statement(statement):
    """Create a slight variation of a statement.
    
    Args:
        statement: Original statement
        
    Returns:
        Modified statement
    """
    modifiers = [
        "Based on recent evidence, ", 
        "According to clinical guidelines, ",
        "Research suggests that ",
        "Studies indicate that ",
        "Clinical practice shows that ",
        "Experts recommend that ",
        "Evidence supports that ",
        "It is widely accepted that ",
        "Current practice indicates that ",
        "Medical consensus suggests that "
    ]
    
    qualifiers = [
        " in most cases",
        " for most patients",
        " in clinical settings",
        " when properly administered",
        " under medical supervision",
        " with appropriate monitoring",
        " in the absence of contraindications",
        " in recommended doses",
        " following proper assessment",
        " as part of a comprehensive treatment plan"
    ]
    
    # Apply modifications
    modified = statement
    
    # 50% chance to add a prefix
    if random.random() < 0.5:
        modified = random.choice(modifiers) + modified.lower()
    
    # 30% chance to add a suffix
    if random.random() < 0.3:
        modified = modified.rstrip(".") + random.choice(qualifiers) + "."
    
    return modified

def demonstrate_continuous_learning(data_dir="data/healthcare", examples_to_generate=20, accuracy=0.75):
    """Run a demonstration of the continuous learning system.
    
    Args:
        data_dir: Directory for healthcare data
        examples_to_generate: Number of examples to generate
        accuracy: Simulated model accuracy
    """
    console.print("\n[bold blue]Healthcare Contradiction Detection - Continuous Learning Demo[/bold blue]\n")
    
    # Step 1: Generate synthetic evaluation results
    console.print("[bold]Step 1: Generating synthetic evaluation results[/bold]")
    eval_dir = os.path.join(data_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    eval_path = os.path.join(eval_dir, "synthetic_results.json")
    
    generate_synthetic_evaluation(eval_path, accuracy)
    
    # Step 2: Initialize continuous learning system
    console.print("\n[bold]Step 2: Initializing continuous learning system[/bold]")
    
    # Use the specific evaluation results file path, not just the directory
    try:
        learner = HealthcareContinuousLearning(
            data_dir=data_dir,
            metrics_dir=eval_path
        )
    except Exception as e:
        logger.error(f"Failed to initialize continuous learning: {str(e)}")
        
        # Simplified approach - directly load the evaluation results
        console.print("[yellow]Using direct approach instead of HealthcareContinuousLearning class[/yellow]")
        with open(eval_path, 'r') as f:
            evaluation_results = json.load(f)
            
        # Create a simplified analysis of the results
        analysis = analyze_performance(evaluation_results, 
                                      improvement_threshold=0.10)
        
        # Display improvement areas
        improvement_areas = analysis.get("improvement_areas", [])
        if improvement_areas:
            table = Table(title="Improvement Areas")
            table.add_column("Type", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Current Accuracy", style="yellow")
            table.add_column("Target Accuracy", style="blue")
            
            for area in improvement_areas:
                table.add_row(
                    area.get("type", ""),
                    area.get("name", ""),
                    f"{area.get('current_accuracy', 0):.2f}",
                    f"{area.get('target_accuracy', 0):.2f}"
                )
            
            console.print(table)
        else:
            console.print("[yellow]No improvement areas identified[/yellow]")
        
        # Generate examples directly
        console.print("\n[bold]Step 3: Generating new training examples[/bold]")
        new_examples = generate_examples(analysis, examples_to_generate)
        
        console.print(f"Generated {len(new_examples)} new training examples")
        
        if new_examples:
            sample_size = min(3, len(new_examples))
            table = Table(title=f"Sample of New Training Examples (showing {sample_size} of {len(new_examples)})")
            table.add_column("Statement 1", style="green", max_width=50)
            table.add_column("Statement 2", style="yellow", max_width=50)
            table.add_column("Type", style="cyan")
            table.add_column("Domain", style="blue")
            
            for i in range(sample_size):
                example = new_examples[i]
                table.add_row(
                    example.get("statement1", "")[:50] + "..." if len(example.get("statement1", "")) > 50 else example.get("statement1", ""),
                    example.get("statement2", "")[:50] + "..." if len(example.get("statement2", "")) > 50 else example.get("statement2", ""),
                    example.get("type", ""),
                    example.get("domain", "")
                )
            
            console.print(table)
        
        # Update training data directly
        console.print("\n[bold]Step 4: Updating training data[/bold]")
        training_path = os.path.join(data_dir, "training", "healthcare_training.json")
        if os.path.exists(training_path):
            with open(training_path, 'r') as f:
                try:
                    existing_examples = json.load(f)
                except json.JSONDecodeError:
                    existing_examples = []
        else:
            existing_examples = []
            
        # Combine existing and new examples
        updated_examples = existing_examples + new_examples
        
        # Save updated training data
        os.makedirs(os.path.dirname(training_path), exist_ok=True)
        with open(training_path, 'w') as f:
            json.dump(updated_examples, f, indent=2)
            
        console.print(f"[green]Updated training data at: {training_path}[/green]")
        
        # Track learning event manually
        history_path = os.path.join(data_dir, "learning_history.json")
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                try:
                    history = json.load(f)
                except json.JSONDecodeError:
                    history = {"events": [], "metrics": {}}
        else:
            history = {"events": [], "metrics": {}}
            
        # Add new event
        history["events"].append({
            "timestamp": datetime.now().isoformat(),
            "type": "training_update",
            "metrics": {
                "examples_generated": len(new_examples),
                "improvement_areas": len(analysis.get("improvement_areas", [])),
                "current_accuracy": evaluation_results.get("accuracy", 0)
            }
        })
        
        # Save updated history
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
            
        console.print("[green]Tracked learning event in history[/green]")
        
        # Add summary
        console.print("\n[bold blue]Continuous Learning Cycle Summary[/bold blue]")
        console.print(f"Starting accuracy: {evaluation_results.get('accuracy', 0):.2f}")
        console.print(f"Improvement areas identified: {len(improvement_areas)}")
        console.print(f"New examples generated: {len(new_examples)}")
        console.print(f"Current training dataset size: {len(updated_examples)}")
        console.print("\n[bold green]Continuous learning cycle completed successfully![/bold green]")
        
        # Return results
        return {
            "improvement_areas": len(analysis.get("improvement_areas", [])),
            "new_examples": len(new_examples),
            "training_path": training_path
        }
    
    # Load the evaluation results
    with open(eval_path, 'r') as f:
        evaluation_results = json.load(f)
    
    # Step 3: Analyze performance
    console.print("\n[bold]Step 3: Analyzing model performance[/bold]")
    analysis = learner.analyze_performance(evaluation_results)
    
    # Display improvement areas
    improvement_areas = analysis.get("improvement_areas", [])
    if improvement_areas:
        table = Table(title="Improvement Areas")
        table.add_column("Type", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Current Accuracy", style="yellow")
        table.add_column("Target Accuracy", style="blue")
        
        for area in improvement_areas:
            table.add_row(
                area.get("type", ""),
                area.get("name", ""),
                f"{area.get('current_accuracy', 0):.2f}",
                f"{area.get('target_accuracy', 0):.2f}"
            )
        
        console.print(table)
    else:
        console.print("[yellow]No improvement areas identified[/yellow]")
    
    # Step 4: Generate new training examples
    console.print("\n[bold]Step 4: Generating new training examples[/bold]")
    new_examples = learner.generate_training_examples(analysis, examples_to_generate)
    
    console.print(f"Generated {len(new_examples)} new training examples")
    
    if new_examples:
        sample_size = min(3, len(new_examples))
        table = Table(title=f"Sample of New Training Examples (showing {sample_size} of {len(new_examples)})")
        table.add_column("Statement 1", style="green", max_width=50)
        table.add_column("Statement 2", style="yellow", max_width=50)
        table.add_column("Type", style="cyan")
        table.add_column("Domain", style="blue")
        
        for i in range(sample_size):
            example = new_examples[i]
            table.add_row(
                example.get("statement1", "")[:50] + "..." if len(example.get("statement1", "")) > 50 else example.get("statement1", ""),
                example.get("statement2", "")[:50] + "..." if len(example.get("statement2", "")) > 50 else example.get("statement2", ""),
                example.get("type", ""),
                example.get("domain", "")
            )
        
        console.print(table)
    
    # Step 5: Update training data
    console.print("\n[bold]Step 5: Updating training data[/bold]")
    training_path = learner.update_training_data(new_examples)
    
    if training_path:
        console.print(f"[green]Successfully updated training data at: {training_path}[/green]")
        
        # Count total examples
        try:
            with open(training_path, 'r') as f:
                total_examples = len(json.load(f))
            console.print(f"Total training examples now available: {total_examples}")
        except Exception as e:
            logger.error(f"Error counting examples: {str(e)}")
    else:
        console.print("[red]Failed to update training data[/red]")
    
    # Step 6: Track learning event
    console.print("\n[bold]Step 6: Tracking learning events[/bold]")
    learner.track_learning_event("training_update", {
        "examples_generated": len(new_examples),
        "improvement_areas": len(improvement_areas),
        "current_accuracy": evaluation_results.get("accuracy", 0)
    })
    
    console.print("[green]Tracked learning event in history[/green]")
    
    # Summary
    console.print("\n[bold blue]Continuous Learning Cycle Summary[/bold blue]")
    console.print(f"Starting accuracy: {evaluation_results.get('accuracy', 0):.2f}")
    console.print(f"Improvement areas identified: {len(improvement_areas)}")
    console.print(f"New examples generated: {len(new_examples)}")
    console.print("\n[bold green]Continuous learning cycle completed successfully![/bold green]")
    
    return {
        "improvement_areas": len(improvement_areas),
        "new_examples": len(new_examples),
        "training_path": training_path
    }

def main():
    parser = argparse.ArgumentParser(description="Demonstrate healthcare continuous learning")
    parser.add_argument("--data-dir", 
                       type=str, 
                       default="data/healthcare",
                       help="Path to healthcare data directory")
    parser.add_argument("--examples", 
                       type=int, 
                       default=20,
                       help="Number of new examples to generate")
    parser.add_argument("--accuracy", 
                       type=float, 
                       default=0.75,
                       help="Simulated model accuracy")
    args = parser.parse_args()
    
    demonstrate_continuous_learning(
        data_dir=args.data_dir,
        examples_to_generate=args.examples,
        accuracy=args.accuracy
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
