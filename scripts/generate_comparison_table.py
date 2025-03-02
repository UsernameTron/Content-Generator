#!/usr/bin/env python3
"""
Generate a comparison table in HTML that shows metrics from previous and current evaluations.
"""
import json
import os
import sys
from datetime import datetime

# Sample data from PDF evaluation (these would be replaced with actual values from the PDF)
pdf_metrics = {
    "customer_experience": {
        "overall_score": 0.81,
        "response_time": 0.85,
        "satisfaction": 0.79,
        "usability": 0.82
    },
    "artificial_intelligence": {
        "overall_score": 0.76,
        "reasoning": 0.74,
        "knowledge_integration": 0.78,
        "adaptability": 0.75
    },
    "machine_learning": {
        "overall_score": 0.78,
        "prediction_accuracy": 0.80,
        "model_robustness": 0.76,
        "generalization": 0.77
    },
    "cross_reference": {
        "overall_score": 0.72,
        "consistency": 0.71,
        "completeness": 0.73,
        "relevance": 0.74
    },
    "counterfactual_reasoning": {
        "overall_score": 0.69,
        "plausibility": 0.67,
        "coherence": 0.70,
        "relevance": 0.71
    },
    "healthcare": {
        "contradiction_detection": {
            "accuracy": 0.79,
            "precision": 0.82,
            "recall": 0.76,
            "f1_score": 0.79
        }
    }
}

def generate_comparison_table(current_data_file, output_file):
    """Generate HTML comparison table between PDF metrics and current metrics."""
    # Load current evaluation data
    with open(current_data_file, 'r') as f:
        current_metrics = json.load(f)
    
    # Create HTML content
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Metrics Comparison Table</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            h1 {
                text-align: center;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
                margin-bottom: 30px;
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
            .domain-header {
                background-color: #2c3e50;
                color: white;
                font-weight: bold;
                text-align: left;
            }
            .metric-name {
                text-align: left;
                padding-left: 30px;
            }
            .positive-change {
                color: #27ae60;
                font-weight: bold;
            }
            .negative-change {
                color: #e74c3c;
                font-weight: bold;
            }
            .no-change {
                color: #7f8c8d;
            }
            .footer {
                text-align: center;
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                font-size: 0.9em;
                color: #7f8c8d;
            }
        </style>
    </head>
    <body>
        <h1>Metrics Comparison: Previous vs Current Evaluation</h1>
        
        <table>
            <thead>
                <tr>
                    <th>Domain / Metric</th>
                    <th>Previous Evaluation</th>
                    <th>Current Evaluation</th>
                    <th>Difference</th>
                </tr>
            </thead>
            <tbody>
    """
    
    # Add rows for each domain and metric
    for domain in ["customer_experience", "artificial_intelligence", "machine_learning", 
                  "cross_reference", "counterfactual_reasoning", "healthcare"]:
        
        # Add domain header
        domain_display = domain.replace('_', ' ').title()
        html += f"""
                <tr class="domain-header">
                    <td colspan="4">{domain_display}</td>
                </tr>
        """
        
        # Add metrics for the domain
        if domain == "healthcare":
            # Special handling for healthcare domain
            if "contradiction_detection" in pdf_metrics.get(domain, {}) and "contradiction_detection" in current_metrics.get(domain, {}):
                pdf_cd = pdf_metrics[domain]["contradiction_detection"]
                current_cd = current_metrics[domain]["contradiction_detection"]
                
                for metric in ["accuracy", "precision", "recall", "f1_score"]:
                    if metric in pdf_cd and metric in current_cd:
                        pdf_value = pdf_cd[metric]
                        current_value = current_cd[metric]
                        diff = current_value - pdf_value
                        
                        # Format the difference with color based on positive/negative
                        if diff > 0:
                            diff_html = f'<span class="positive-change">+{diff:.2f}</span>'
                        elif diff < 0:
                            diff_html = f'<span class="negative-change">{diff:.2f}</span>'
                        else:
                            diff_html = f'<span class="no-change">0.00</span>'
                        
                        html += f"""
                        <tr>
                            <td class="metric-name">Contradiction {metric.replace('_', ' ').title()}</td>
                            <td>{pdf_value:.2f}</td>
                            <td>{current_value:.2f}</td>
                            <td>{diff_html}</td>
                        </tr>
                        """
        else:
            # Standard domain metrics
            pdf_domain = pdf_metrics.get(domain, {})
            current_domain = current_metrics.get(domain, {})
            
            for metric in ["overall_score", "response_time", "satisfaction", "usability", 
                          "reasoning", "knowledge_integration", "adaptability",
                          "prediction_accuracy", "model_robustness", "generalization",
                          "consistency", "completeness", "relevance",
                          "plausibility", "coherence"]:
                
                if metric in pdf_domain and metric in current_domain:
                    pdf_value = pdf_domain[metric]
                    current_value = current_domain[metric]
                    diff = current_value - pdf_value
                    
                    # Format the difference with color based on positive/negative
                    if diff > 0:
                        diff_html = f'<span class="positive-change">+{diff:.2f}</span>'
                    elif diff < 0:
                        diff_html = f'<span class="negative-change">{diff:.2f}</span>'
                    else:
                        diff_html = f'<span class="no-change">0.00</span>'
                    
                    metric_display = metric.replace('_', ' ').title()
                    html += f"""
                    <tr>
                        <td class="metric-name">{metric_display}</td>
                        <td>{pdf_value:.2f}</td>
                        <td>{current_value:.2f}</td>
                        <td>{diff_html}</td>
                    </tr>
                    """
    
    # Add summary section
    pdf_summary = pdf_metrics.get("summary", {})
    current_summary = current_metrics.get("summary", {})
    
    if "overall_performance" in current_summary:
        pdf_overall = pdf_summary.get("overall_performance", 0.75)  # Default if not available
        current_overall = current_summary["overall_performance"]
        diff = current_overall - pdf_overall
        
        # Format the difference
        if diff > 0:
            diff_html = f'<span class="positive-change">+{diff:.2f}</span>'
        elif diff < 0:
            diff_html = f'<span class="negative-change">{diff:.2f}</span>'
        else:
            diff_html = f'<span class="no-change">0.00</span>'
        
        html += f"""
                <tr class="domain-header">
                    <td colspan="4">Overall Summary</td>
                </tr>
                <tr>
                    <td class="metric-name">Overall Performance</td>
                    <td>{pdf_overall:.2f}</td>
                    <td>{current_overall:.2f}</td>
                    <td>{diff_html}</td>
                </tr>
        """
    
    html += """
            </tbody>
        </table>
        
        <div class="footer">
            <p>Generated on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """ | Healthcare Metrics Visualization System</p>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"Comparison table generated: {output_file}")
    return output_file

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_comparison_table.py <current_data_file> <output_file>")
        sys.exit(1)
    
    current_data_file = sys.argv[1]
    output_file = sys.argv[2]
    
    generate_comparison_table(current_data_file, output_file)
