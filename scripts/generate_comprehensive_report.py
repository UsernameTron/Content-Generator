#!/usr/bin/env python3
"""
Generate a comprehensive HTML report showing performance benchmarks and improvements
across all three enhancement areas: contradiction detection, counterfactual reasoning,
and cross-referencing capabilities.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

def load_enhancement_data(enhancement_dir):
    """Load data from all three enhancement plan JSON files."""
    enhancement_files = {
        "contradiction_detection": "contradiction_enhancement_plan.json",
        "counterfactual_reasoning": "counterfactual_enhancement_plan.json",
        "cross_reference": "cross_reference_enhancement_plan.json",
        "customer_experience": "customer_experience_enhancement_plan.json",
        "artificial_intelligence": "artificial_intelligence_enhancement_plan.json"
    }
    
    data = {}
    for area, filename in enhancement_files.items():
        file_path = os.path.join(enhancement_dir, filename)
        try:
            with open(file_path, 'r') as f:
                data[area] = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Enhancement file {file_path} not found.")
    
    return data

def generate_comprehensive_report(enhancement_data, output_file):
    """Generate comprehensive HTML report with data from all enhancement areas."""
    
    # Start generating HTML
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Healthcare Performance Enhancement System - Comprehensive Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        h1 {
            text-align: center;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }
        .container {
            display: flex;
            flex-direction: column;
            gap: 30px;
        }
        .metric-container {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .metric-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .metric-title {
            font-size: 1.4em;
            font-weight: bold;
            margin: 0;
        }
        .metric-score {
            font-size: 1.2em;
            font-weight: bold;
        }
        .progress-container {
            margin-bottom: 20px;
        }
        .progress-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        .progress-bar-container {
            height: 24px;
            background-color: #ecf0f1;
            border-radius: 12px;
            overflow: hidden;
            position: relative;
        }
        .progress-bar {
            height: 100%;
            background-color: #3498db;
            border-radius: 12px;
            transition: width 1s ease-in-out;
        }
        .target-marker {
            position: absolute;
            top: 0;
            height: 100%;
            width: 3px;
            background-color: #e74c3c;
        }
        .target-label {
            position: absolute;
            top: -20px;
            transform: translateX(-50%);
            font-size: 0.8em;
            color: #e74c3c;
        }
        .intervention {
            background-color: #fff;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #3498db;
        }
        .intervention h3 {
            margin-top: 0;
            color: #3498db;
        }
        .intervention-impact {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }
        .impact-item {
            background-color: #eef7fe;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.9em;
        }
        .footer {
            text-align: center;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            font-size: 0.9em;
            color: #7f8c8d;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px 15px;
            border: 1px solid #ddd;
            text-align: center;
        }
        th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .summary-table th:first-child,
        .summary-table td:first-child {
            text-align: left;
        }
        .positive-change {
            color: #27ae60;
            font-weight: bold;
        }
        .system-specs {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin-top: 30px;
            border-left: 4px solid #9b59b6;
        }
        .system-specs h2 {
            margin-top: 0;
            color: #9b59b6;
        }
        .spec-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .spec-item {
            background-color: #fff;
            padding: 10px 15px;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .spec-item strong {
            display: block;
            margin-bottom: 5px;
            color: #2c3e50;
        }
    </style>
</head>
<body>
    <h1>Healthcare Performance Enhancement System<br>Comprehensive Report</h1>
    
    <!-- Summary Table -->
    <div class="metric-container">
        <h2>Performance Enhancement Summary</h2>
        <table class="summary-table">
            <thead>
                <tr>
                    <th>Enhancement Area</th>
                    <th>Baseline Score</th>
                    <th>Target Score</th>
                    <th>Projected Score</th>
                    <th>Improvement</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # Add summary table rows
    for area, data in enhancement_data.items():
        area_name = area.replace("_", " ").title()
        
        baseline = 0
        target = 0
        projected = 0
        
        if area == "contradiction_detection":
            if "baseline_metrics" in data and "overall_metrics" in data["baseline_metrics"]:
                baseline = data["baseline_metrics"]["overall_metrics"].get("f1_score", 0)
            if "enhancement_targets" in data and "target_metrics" in data["enhancement_targets"]:
                target = data["enhancement_targets"]["target_metrics"].get("f1_score", 0)
            if "enhancement_targets" in data and "projected_metrics" in data["enhancement_targets"]:
                projected = data["enhancement_targets"]["projected_metrics"].get("f1_score", 0)
                
        elif area == "customer_experience" or area == "artificial_intelligence":
            if "baseline_metrics" in data:
                baseline = data["baseline_metrics"].get("overall_score", 0)
            if "enhancement_targets" in data and "target_metrics" in data["enhancement_targets"]:
                target = data["enhancement_targets"]["target_metrics"].get("overall_score", 0)
            if "enhancement_targets" in data and "projected_metrics" in data["enhancement_targets"]:
                projected = data["enhancement_targets"]["projected_metrics"].get("overall_score", 0)
                
        elif area == "counterfactual_reasoning":
            if "baseline_metrics" in data:
                baseline = data["baseline_metrics"].get("overall_score", 0)
            if "enhancement_targets" in data and "target_metrics" in data["enhancement_targets"]:
                target = data["enhancement_targets"]["target_metrics"].get("overall_score", 0)
            if "enhancement_targets" in data and "projected_metrics" in data["enhancement_targets"]:
                projected = data["enhancement_targets"]["projected_metrics"].get("overall_score", 0)
                
        elif area == "cross_reference":
            if "baseline_metrics" in data:
                baseline = data["baseline_metrics"].get("overall_score", 0)
            if "enhancement_targets" in data and "target_metrics" in data["enhancement_targets"]:
                target = data["enhancement_targets"]["target_metrics"].get("overall_score", 0)
            if "enhancement_targets" in data and "projected_metrics" in data["enhancement_targets"]:
                projected = data["enhancement_targets"]["projected_metrics"].get("overall_score", 0)
        
        improvement = projected - baseline
        improvement_class = "positive-change" if improvement > 0 else ""
        
        html += f"""
                <tr>
                    <td>{area_name}</td>
                    <td>{baseline:.2f}</td>
                    <td>{target:.2f}</td>
                    <td>{projected:.2f}</td>
                    <td class="{improvement_class}">+{improvement:.2f}</td>
                </tr>
        """
    
    html += """
            </tbody>
        </table>
    </div>
    
    <!-- Detailed Sections -->
    <div class="container">
    """
    
    # Process each enhancement area
    for area, data in enhancement_data.items():
        area_title = area.replace("_", " ").title()
        
        # Determine the main metrics for this area
        if area == "contradiction_detection":
            main_metrics = {
                "accuracy": "Accuracy",
                "precision": "Precision",
                "recall": "Recall",
                "f1_score": "F1 Score"
            }
            baseline_data = data.get("baseline_metrics", {}).get("overall_metrics", {})
            target_data = data.get("enhancement_targets", {}).get("target_metrics", {})
            projected_data = data.get("enhancement_targets", {}).get("projected_metrics", {})
            
        elif area == "counterfactual_reasoning":
            main_metrics = {
                "overall_score": "Overall Score",
                "plausibility": "Plausibility",
                "coherence": "Coherence"
            }
            baseline_data = data.get("baseline_metrics", {})
            target_data = data.get("enhancement_targets", {}).get("target_metrics", {})
            projected_data = data.get("enhancement_targets", {}).get("projected_metrics", {})
            
        elif area == "cross_reference":
            main_metrics = {
                "overall_score": "Overall Score",
                "consistency": "Consistency",
                "completeness": "Completeness",
                "relevance": "Relevance"
            }
            baseline_data = data.get("baseline_metrics", {})
            target_data = data.get("enhancement_targets", {}).get("target_metrics", {})
            projected_data = data.get("enhancement_targets", {}).get("projected_metrics", {})
            
        elif area == "customer_experience":
            main_metrics = {
                "overall_score": "Overall Score",
                "response_time": "Response Time",
                "satisfaction": "Satisfaction",
                "usability": "Usability"
            }
            baseline_data = data.get("baseline_metrics", {})
            target_data = data.get("enhancement_targets", {}).get("target_metrics", {})
            projected_data = data.get("enhancement_targets", {}).get("projected_metrics", {})
            
        elif area == "artificial_intelligence":
            main_metrics = {
                "overall_score": "Overall Score",
                "reasoning": "Reasoning",
                "knowledge_integration": "Knowledge Integration",
                "adaptability": "Adaptability"
            }
            baseline_data = data.get("baseline_metrics", {})
            target_data = data.get("enhancement_targets", {}).get("target_metrics", {})
            projected_data = data.get("enhancement_targets", {}).get("projected_metrics", {})
        
        # Add section container
        html += f"""
        <div class="metric-container">
            <div class="metric-header">
                <h2 class="metric-title">{area_title}</h2>
            </div>
        """
        
        # Add metrics visualization
        for metric_key, metric_name in main_metrics.items():
            baseline_value = baseline_data.get(metric_key, 0)
            target_value = target_data.get(metric_key, 0)
            projected_value = projected_data.get(metric_key, 0)
            
            # Calculate percentages for rendering
            baseline_percent = baseline_value * 100
            target_percent = target_value * 100
            projected_percent = projected_value * 100
            
            html += f"""
            <div class="progress-container">
                <div class="progress-label">
                    <span>{metric_name}</span>
                    <span>Baseline: {baseline_value:.2f} → Projected: {projected_value:.2f}</span>
                </div>
                <div class="progress-bar-container">
                    <div class="progress-bar" style="width: {baseline_percent}%;"></div>
                    <div class="progress-bar" style="width: {projected_percent}%; background-color: rgba(52, 152, 219, 0.5);"></div>
                    <div class="target-marker" style="left: {target_percent}%;">
                        <span class="target-label">Target: {target_value:.2f}</span>
                    </div>
                </div>
            </div>
            """
        
        # Add interventions
        if "enhancement_targets" in data and "interventions" in data["enhancement_targets"]:
            html += "<h3>Targeted Interventions</h3>"
            
            for intervention in data["enhancement_targets"]["interventions"]:
                name = intervention.get("name", "Untitled")
                description = intervention.get("description", "No description")
                
                html += f"""
                <div class="intervention">
                    <h3>{name.replace("_", " ").title()}</h3>
                    <p>{description}</p>
                    <div class="intervention-impact">
                """
                
                if "expected_improvement" in intervention:
                    for metric, value in intervention["expected_improvement"].items():
                        html += f"""
                        <div class="impact-item">
                            {metric.replace("_", " ").title()}: +{value:.2f}
                        </div>
                        """
                
                html += """
                    </div>
                </div>
                """
        
        html += "</div>"
    
    # Add hardware specs section
    html += """
    <div class="system-specs">
        <h2>Hardware Optimization</h2>
        <div class="spec-grid">
            <div class="spec-item">
                <strong>CPU</strong>
                Apple M4 Pro (12-core)
            </div>
            <div class="spec-item">
                <strong>RAM</strong>
                48GB Unified Memory
            </div>
            <div class="spec-item">
                <strong>Storage</strong>
                512GB NVMe SSD
            </div>
            <div class="spec-item">
                <strong>GPU</strong>
                18-core Apple GPU
            </div>
            <div class="spec-item">
                <strong>Neural Engine</strong>
                16-core
            </div>
        </div>
    </div>
    """
    
    # Close HTML
    html += f"""
    </div>
    
    <div class="footer">
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Healthcare Performance Enhancement System</p>
    </div>
</body>
</html>
    """
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"Comprehensive report generated: {output_file}")

def main():
    if len(sys.argv) < 3:
        print("Usage: generate_comprehensive_report.py <enhancement_data_dir> <output_file>")
        sys.exit(1)
    
    enhancement_dir = sys.argv[1]
    output_file = sys.argv[2]
    
    # Load data
    enhancement_data = load_enhancement_data(enhancement_dir)
    
    # Generate report
    generate_comprehensive_report(enhancement_data, output_file)

if __name__ == "__main__":
    main()
