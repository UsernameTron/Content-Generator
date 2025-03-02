#!/usr/bin/env python3
"""
Dataset Enhancement Script for C. Pete Connor Fine-tuning

This script creates higher-quality examples for fine-tuning by:
1. Ensuring consistent format with proper chat tokens
2. Generating examples across multiple content categories
3. Emphasizing direct, professional responses without "meta-commentary"
"""

import os
import json
import argparse
from rich.console import Console
from rich.progress import Progress

console = Console()

# Default output directory
OUTPUT_DIR = "dataset"

# Sample high-quality examples across different categories
SAMPLE_EXAMPLES = [
    # Marketing Content
    {
        "prompt": "Write a social media post for a new cafe focusing on organic ingredients.",
        "response": "Introducing Harvest Cafe: Where every bite tells a story of sustainability and flavor! 🌱 Our newly opened doors welcome you to experience coffee and cuisine crafted from 100% organic, locally-sourced ingredients. From farm-fresh breakfast bowls to artisanal pastries paired with specialty brews, we're redefining wholesome indulgence. This weekend only: Mention this post for 15% off your first visit! #OrganicLiving #FarmToTable #HarvestCafe"
    },
    {
        "prompt": "Create a product description for wireless headphones highlighting noise cancellation.",
        "response": "SonicPro X300: Immerse Yourself in Pure Sound\n\nEscape into your personal audio sanctuary with our premium SonicPro X300 wireless headphones. Featuring industry-leading adaptive noise cancellation technology that dynamically responds to your environment, these headphones intelligently neutralize up to 98% of external noise while preserving audio clarity.\n\nKey features:\n• 40-hour battery life with quick-charge capability (10 min charge = 5 hours playback)\n• Bluetooth 5.2 with multipoint connection for seamless device switching\n• Memory foam ear cushions with breathable protein leather covering\n• Voice-assistant compatible with dedicated control button\n• IPX4 water resistance for worry-free workouts\n\nExperience sound as it was meant to be heard, without compromise."
    },
    
    # Email & Communication
    {
        "prompt": "Draft a professional email requesting a meeting with a potential client.",
        "response": "Subject: Request for Meeting - Innovative Solutions for [Company Name]\n\nDear [Name],\n\nI hope this email finds you well. My name is [Your Name], [Your Position] at [Your Company].\n\nI've been following [Client's Company]'s recent developments in [relevant industry area] and was particularly impressed by your [mention specific achievement or project].\n\nI believe there are significant opportunities for collaboration that could benefit both our organizations. Our [mention your relevant service/product] has helped similar companies achieve [mention specific results] and I'd welcome the chance to discuss how we might tailor our approach to your specific needs.\n\nWould you be available for a 30-minute call next week? I'm flexible on Tuesday or Thursday between 10am-4pm, but can certainly work around your schedule if those times don't suit.\n\nThank you for considering my request. I look forward to potentially speaking with you.\n\nBest regards,\n\n[Your Name]\n[Your Position]\n[Your Company]\n[Contact Information]"
    },
    
    # Technical Content
    {
        "prompt": "Explain the concept of API rate limiting to a non-technical audience.",
        "response": "API Rate Limiting: Think of it Like a Coffee Shop\n\nImagine you run a busy coffee shop with limited staff. You can serve a certain number of customers per hour efficiently, but if too many people show up at once, the quality of service suffers, lines grow longer, and eventually, you might have to turn people away.\n\nAPI rate limiting works similarly for digital services. When you use an app on your phone or a website, that app makes requests to a digital service (called an API) to fetch information or perform actions. The company providing this service needs to manage how many requests their systems handle to ensure everyone gets good service.\n\nRate limiting is their way of saying: \"You can make X number of requests per minute/hour/day.\" If an app tries to make too many requests too quickly, the service temporarily blocks additional requests with a polite \"please wait a bit\" message.\n\nThis system protects the service from being overwhelmed, prevents any single user from monopolizing resources, and helps maintain reliability for everyone. It's like the coffee shop ensuring that one customer doesn't order 100 drinks at once while others wait.\n\nFor everyday users, you rarely notice rate limits unless you're doing something unusual. For developers building apps, they need to work within these limits by spacing out requests or prioritizing the most important ones."
    },
    
    # Business Writing
    {
        "prompt": "Write a brief project update for stakeholders on a software development project that's running behind schedule.",
        "response": "Project Falcon: Mid-Quarter Status Update\n\nDear Stakeholders,\n\nI'm writing to provide an update on Project Falcon's progress and to address our current timeline adjustments.\n\n**Current Status:**\nWe've successfully completed the database migration and security infrastructure components, which passed all compliance requirements. The user authentication system is currently at 90% completion and undergoing final testing.\n\n**Timeline Adjustment:**\nAs you may be aware, we're currently tracking 2 weeks behind our original delivery date. This delay stems from two primary factors:\n\n1. The third-party payment processing API integration revealed unexpected compatibility issues requiring additional development work.\n2. Our lead back-end developer was unexpectedly absent for 10 days due to a family emergency.\n\n**Mitigation Plan:**\nTo address these challenges, we have:\n\n1. Reassigned two developers from non-critical path tasks to assist with the payment integration\n2. Implemented extended development hours (voluntary, with compensation)\n3. Reprioritized the feature roadmap to ensure critical elements launch on time\n\n**Revised Timeline:**\nWith these adjustments, we project delivery of core functionality by March 15th, with secondary features to follow in a phased release approach by March 29th.\n\n**Next Steps:**\nA detailed revised project plan will be distributed by EOD Friday. I welcome any questions or concerns you might have during our weekly stakeholder call tomorrow at 11am.\n\nThank you for your continued support as we work toward delivering a robust and secure platform.\n\nSincerely,\n[Project Manager]"
    }
]

def format_dataset(examples, output_file):
    """Format examples into the proper chat format for fine-tuning"""
    formatted_examples = []
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Formatting examples...", total=len(examples))
        
        for example in examples:
            formatted_example = {
                "text": f"<|prompter|>{example['prompt']}<|assistant|>{example['response']}"
            }
            formatted_examples.append(formatted_example)
            progress.update(task, advance=1)
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write the formatted examples to a JSON file
    with open(output_file, 'w') as f:
        json.dump(formatted_examples, f, indent=2)
    
    console.print(f"[bold green]Successfully created {len(formatted_examples)} formatted examples.")
    console.print(f"[bold]Output file:[/bold] {output_file}")

def create_jsonl_file(examples, output_file):
    """Create a JSONL file for compatibility with more training scripts"""
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        for example in examples:
            formatted_example = {
                "text": f"<|prompter|>{example['prompt']}<|assistant|>{example['response']}"
            }
            f.write(json.dumps(formatted_example) + '\n')
    
    console.print(f"[bold green]Successfully created JSONL file with {len(examples)} examples.")
    console.print(f"[bold]Output file:[/bold] {output_file}")

def enhance_existing_dataset(input_file, output_file):
    """Enhance an existing dataset with proper formatting"""
    console.print(f"[bold]Enhancing existing dataset:[/bold] {input_file}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        console.print(f"[bold red]Error:[/bold red] Input file not found: {input_file}")
        return
    
    # Read the input file
    with open(input_file, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            # Try to read as JSONL
            f.seek(0)
            data = []
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    console.print(f"[bold red]Error:[/bold red] Invalid JSON format in line: {line}")
    
    # Extract examples from the input data
    examples = []
    
    if isinstance(data, list):
        for item in data:
            if "text" in item:
                # Extract prompt and response from text field
                text = item["text"]
                if "<|prompter|>" in text and "<|assistant|>" in text:
                    parts = text.split("<|assistant|>")
                    if len(parts) >= 2:
                        prompt = parts[0].replace("<|prompter|>", "").strip()
                        response = "<|assistant|>".join(parts[1:]).strip()
                        examples.append({"prompt": prompt, "response": response})
            elif "prompt" in item and "response" in item:
                examples.append({"prompt": item["prompt"], "response": item["response"]})
    
    # Add our sample examples
    examples.extend(SAMPLE_EXAMPLES)
    
    # Format and save the enhanced dataset
    format_dataset(examples, output_file)

def main():
    parser = argparse.ArgumentParser(description="Dataset enhancement for C. Pete Connor model fine-tuning")
    parser.add_argument("--input", type=str, help="Input dataset file (JSON or JSONL)")
    parser.add_argument("--output", type=str, default=os.path.join(OUTPUT_DIR, "enhanced_dataset.json"),
                        help="Output file path for the enhanced dataset")
    parser.add_argument("--jsonl", action="store_true", help="Also create a JSONL version of the dataset")
    args = parser.parse_args()
    
    console.rule("[bold blue]C. Pete Connor Model - Dataset Enhancement")
    
    if args.input:
        enhance_existing_dataset(args.input, args.output)
    else:
        console.print("[bold yellow]No input file provided.[/bold yellow] Creating sample dataset...")
        format_dataset(SAMPLE_EXAMPLES, args.output)
    
    if args.jsonl:
        jsonl_output = args.output.replace(".json", ".jsonl")
        if args.input:
            # We already have the examples from enhance_existing_dataset
            with open(args.output, 'r') as f:
                data = json.load(f)
                examples = []
                for item in data:
                    if "text" in item:
                        text = item["text"]
                        if "<|prompter|>" in text and "<|assistant|>" in text:
                            parts = text.split("<|assistant|>")
                            if len(parts) >= 2:
                                prompt = parts[0].replace("<|prompter|>", "").strip()
                                response = "<|assistant|>".join(parts[1:]).strip()
                                examples.append({"prompt": prompt, "response": response})
                create_jsonl_file(examples, jsonl_output)
        else:
            create_jsonl_file(SAMPLE_EXAMPLES, jsonl_output)
    
    console.rule("[bold green]Dataset Enhancement Complete")
    console.print("\n[bold]Next Steps:[/bold]")
    console.print("1. Adjust fine-tuning configuration to use this enhanced dataset")
    console.print("2. Run the training script with the updated dataset path")
    console.print("3. Test the new model with improved prompts and response patterns")

if __name__ == "__main__":
    main()
