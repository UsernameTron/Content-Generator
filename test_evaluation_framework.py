#!/usr/bin/env python3
"""
Test script for the comprehensive evaluation framework.
This runs a lightweight verification of the evaluation components without loading a model.
"""

import os
import sys
import json
import logging
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.logging import RichHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("test_evaluator")

# Set up rich console
console = Console()

def test_evaluation_data():
    """Test that all evaluation data files are present and properly formatted."""
    data_dir = Path("data/evaluation")
    required_files = [
        "customer_experience_questions.json",
        "artificial_intelligence_questions.json",
        "machine_learning_questions.json",
        "cross_reference_scenarios.json",
        "counterfactual_scenarios.json"
    ]
    
    console.print(Panel("[bold]Testing Evaluation Data[/bold]", style="blue"))
    
    all_valid = True
    for filename in required_files:
        file_path = data_dir / filename
        
        # Check if file exists
        if not file_path.exists():
            console.print(f"❌ [bold red]ERROR:[/bold red] File {filename} not found")
            all_valid = False
            continue
        
        try:
            # Load and validate JSON
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, list) or len(data) == 0:
                console.print(f"❌ [bold red]ERROR:[/bold red] {filename} does not contain a list of questions")
                all_valid = False
                continue
                
            # Check first item structure
            first_item = data[0]
            if "question" in first_item and "criteria" in first_item:
                console.print(f"✅ {filename}: {len(data)} questions/scenarios")
            elif "scenario" in first_item and "criteria" in first_item:
                console.print(f"✅ {filename}: {len(data)} questions/scenarios")
            else:
                console.print(f"❌ [bold red]ERROR:[/bold red] {filename} has incorrect format")
                all_valid = False
                
        except json.JSONDecodeError:
            console.print(f"❌ [bold red]ERROR:[/bold red] {filename} is not valid JSON")
            all_valid = False
        except Exception as e:
            console.print(f"❌ [bold red]ERROR:[/bold red] Error processing {filename}: {str(e)}")
            all_valid = False
    
    return all_valid

def test_evaluator_modules():
    """Test that all evaluator modules can be imported and are properly structured."""
    console.print(Panel("[bold]Testing Evaluator Modules[/bold]", style="blue"))
    
    all_valid = True
    
    # Test base evaluator
    try:
        from evaluators import BaseEvaluator
        console.print("✅ BaseEvaluator imported successfully")
    except ImportError as e:
        console.print(f"❌ [bold red]ERROR:[/bold red] BaseEvaluator import failed: {str(e)}")
        all_valid = False
    
    # Test domain knowledge evaluators
    try:
        from evaluators.domain_knowledge import CustomerExperienceEvaluator, ArtificialIntelligenceEvaluator, MachineLearningEvaluator
        console.print("✅ Domain knowledge evaluators imported successfully")
    except ImportError as e:
        console.print(f"❌ [bold red]ERROR:[/bold red] Domain evaluators import failed: {str(e)}")
        all_valid = False
    
    # Test special capability evaluators
    try:
        from evaluators.cross_referencing import CrossReferencingEvaluator
        console.print("✅ CrossReferencingEvaluator imported successfully")
    except ImportError as e:
        console.print(f"❌ [bold red]ERROR:[/bold red] CrossReferencingEvaluator import failed: {str(e)}")
        all_valid = False
        
    try:
        from evaluators.counterfactual import CounterfactualEvaluator
        console.print("✅ CounterfactualEvaluator imported successfully")
    except ImportError as e:
        console.print(f"❌ [bold red]ERROR:[/bold red] CounterfactualEvaluator import failed: {str(e)}")
        all_valid = False
    
    return all_valid

def test_main_script():
    """Test that the main evaluation script is properly structured."""
    console.print(Panel("[bold]Testing Main Evaluation Script[/bold]", style="blue"))
    
    try:
        import comprehensive_evaluate
        console.print("✅ Main evaluation script imported successfully")
        
        # Check for required classes and functions
        if hasattr(comprehensive_evaluate, 'EvaluationManager'):
            console.print("✅ EvaluationManager class found")
        else:
            console.print("❌ [bold red]ERROR:[/bold red] EvaluationManager class not found")
            return False
            
        if hasattr(comprehensive_evaluate, 'parse_args'):
            console.print("✅ Command-line argument parsing found")
        else:
            console.print("❌ [bold red]ERROR:[/bold red] Command-line argument parsing not found")
            return False
        
        return True
        
    except ImportError as e:
        console.print(f"❌ [bold red]ERROR:[/bold red] Main script import failed: {str(e)}")
        return False
    except Exception as e:
        console.print(f"❌ [bold red]ERROR:[/bold red] Error checking main script: {str(e)}")
        return False

def test_desktop_launcher():
    """Test that the desktop launcher script exists and is executable."""
    console.print(Panel("[bold]Testing Desktop Launcher[/bold]", style="blue"))
    
    launcher_path = Path("run_evaluation.command")
    
    if not launcher_path.exists():
        console.print("❌ [bold red]ERROR:[/bold red] Desktop launcher not found")
        return False
        
    if not os.access(launcher_path, os.X_OK):
        console.print("❌ [bold red]ERROR:[/bold red] Desktop launcher is not executable")
        return False
        
    console.print("✅ Desktop launcher exists and is executable")
    return True

def main():
    """Run all tests and report results."""
    console.print(Panel("[bold]C. Pete Connor Model Evaluation Framework Test[/bold]", 
                         style="green", expand=False))
    
    # Run all tests
    data_valid = test_evaluation_data()
    modules_valid = test_evaluator_modules()
    main_script_valid = test_main_script()
    launcher_valid = test_desktop_launcher()
    
    # Report overall status
    console.print("\n")
    console.print(Panel("[bold]Test Results Summary[/bold]", style="blue"))
    
    if data_valid and modules_valid and main_script_valid and launcher_valid:
        console.print("[bold green]✅ ALL TESTS PASSED: Evaluation framework is ready for use[/bold green]")
        return 0
    else:
        console.print("[bold red]❌ SOME TESTS FAILED: Please fix the issues above before running the framework[/bold red]")
        return 1

if __name__ == "__main__":
    sys.exit(main())
