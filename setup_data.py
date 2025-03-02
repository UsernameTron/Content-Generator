"""
Setup script to initialize data files for the project.
"""

import os
import json
import nltk
import logging
import importlib.util
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def setup_data_files():
    """
    Setup necessary data files and NLTK data for the project.
    """
    # Get paths
    project_dir = Path(__file__).parent
    data_dir = project_dir / "data"
    
    # Create data directory if it doesn't exist
    data_dir.mkdir(exist_ok=True)
    logger.info(f"Created data directory: {data_dir}")
    
    # Create training data directory
    training_dir = data_dir / "training"
    training_dir.mkdir(exist_ok=True)
    logger.info(f"Created training data directory: {training_dir}")
    
    # Create outputs directory
    outputs_dir = project_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    # Create finetune output directories
    finetune_dir = outputs_dir / "finetune"
    finetune_dir.mkdir(exist_ok=True)
    finetune_final_dir = finetune_dir / "final"
    finetune_final_dir.mkdir(exist_ok=True)
    logger.info(f"Created model output directories: {finetune_dir}, {finetune_final_dir}")
    
    # Download required NLTK data
    try:
        nltk_packages = ['punkt', 'stopwords', 'vader_lexicon']
        for package in nltk_packages:
            try:
                nltk.data.find(f'tokenizers/{package}' if package != 'vader_lexicon' else f'sentiment/{package}')
                logger.info(f"NLTK package {package} is already downloaded")
            except LookupError:
                nltk.download(package)
                logger.info(f"Downloaded NLTK package: {package}")
        
        print("NLTK data setup completed successfully")
    except Exception as e:
        logger.error(f"Error downloading NLTK data: {str(e)}")
        print(f"Error downloading NLTK data: {str(e)}")
        return False
    
    # Create examples directory for user-uploaded examples
    examples_dir = data_dir / "examples"
    examples_dir.mkdir(exist_ok=True)
    logger.info(f"Created examples directory: {examples_dir}")
    
    # Create writing style file if it doesn't exist
    writing_style_file = data_dir / "writing_style.json"
    if not writing_style_file.exists():
        # Create a starter writing style file for C. Pete Connor
        writing_style = {
            "model_parameters": {
                "name": "pete-connor-satirical-tech-expert",
                "base_model": "mistral-7b",
                "training_objective": "next-token-prediction",
                "temperature": 0.7,
                "max_tokens": 1500,
                "top_p": 0.95,
                "frequency_penalty": 0.2,
                "presence_penalty": 0.1
            },
            "prompt_template": {
                "system_message": "You are C. Pete Connor, a tech industry expert known for your data-driven, no-nonsense approach to cutting through hype and corporate jargon. Your writing combines sharp satire with genuine expertise, using actual statistics and research to expose the gap between corporate promises and reality.",
                "context": "Your audience consists of tech professionals, executives, and industry insiders who appreciate both technical accuracy and well-crafted humor.",
                "task": "Create content that combines skeptical industry analysis with hard data and practical insights."
            },
            "examples": [
                {
                    "description": "A post critiquing AI hype through data",
                    "content": "# 💡 The AI Prediction Game: Are We Moving Too Fast Without Enough Data? 🤔\n\nLet's talk about those magical AI predictions for 2030. Everyone seems to have a crystal ball that shows AI solving world hunger, curing cancer, and probably making the perfect cup of coffee while it's at it.\n\nBut here's what the data actually tells us:\n\n- McKinsey reports that 76% of companies are still struggling with basic data quality issues\n- Only 14% of AI initiatives successfully make it from pilot to production\n- 82% of executives admit their AI investments haven't yet delivered substantial business outcomes\n\nSo while tech bros are painting utopian futures, most organizations can't even get their data lakes in order.\n\nThe disconnect between lofty predictions and actual readiness isn't just amusing—it's dangerous. We're building houses on sand and acting surprised when they collapse.\n\n**One-Liner**: Before promising an AI revolution in 2030, maybe make sure your 2024 data doesn't look like it was collected on stone tablets."
                },
                {
                    "description": "A Twitter post about tech stack complexity",
                    "content": "Your 'modern tech stack' has 47 interconnected services but your developers can't explain how they work together.\n\nNew study: 68% of engineering time is spent maintaining complexity, not adding value.\n\nMaybe the next 10X innovation is just deleting half your codebase. #TechDebt"
                },
                {
                    "description": "A LinkedIn post about startup funding",
                    "content": "🔍 Startup Math That Doesn't Add Up\n\nA startup just raised $50M at a $500M valuation with:\n- $0 in revenue\n- 0 paying customers\n- 1 MVP that's 'launching soon'\n\nInvestors cite their 'innovative AI approach' to a problem that three other funded startups failed to solve last year.\n\nThis isn't investing—it's fantasy football with PowerPoints.\n\n#VCFunding #StartupEconomy"
                }
            ]
        }
        
        # Write the writing style file
        with open(writing_style_file, 'w') as f:
            json.dump(writing_style, f, indent=2)
        
        logger.info(f"Created writing style file at {writing_style_file}")
        print("Created writing style file with C. Pete Connor's satirical tech expert style")
    else:
        logger.info(f"Writing style file already exists at {writing_style_file}")
    
    # Create platform templates file if it doesn't exist
    template_file = data_dir / "custom_templates.json"
    if not template_file.exists():
        # Create a starter template file
        custom_templates = {
            "Twitter": [
                "Just my thoughts on {topic}: {main_point} {hashtags}"
            ],
            "LinkedIn": [
                "I've been thinking about {topic} lately.\n\n{main_point}\n\n{supporting_points}\n\nWhat are your thoughts on this? {hashtags}"
            ],
            "Facebook": [
                "Today's reflection on {topic}...\n\n{main_point}\n\n{supporting_points} {hashtags}"
            ]
        }
        
        # Write the template file
        with open(template_file, 'w') as f:
            json.dump(custom_templates, f, indent=2)
        
        logger.info(f"Created custom templates file at {template_file}")
        print("Created custom templates file with example templates")
    else:
        logger.info(f"Custom templates file already exists at {template_file}")
    
    # Create user preferences file if it doesn't exist
    prefs_file = data_dir / "user_preferences.json"
    if not prefs_file.exists():
        # Create default preferences
        default_prefs = {
            "default_platform": "Twitter",
            "default_tone": "Informative",
            "max_keyword_extract": 10,
            "default_hashtag_count": 3,
            "emoji_usage": "moderate",
            "ui_theme": "light",
            "system": {
                "log_level": "INFO",
                "health_check_interval": 60,
                "cache_processed_docs": True,
                "max_doc_cache_size": 10
            }
        }
        
        # Write the preferences file
        with open(prefs_file, 'w') as f:
            json.dump(default_prefs, f, indent=2)
        
        logger.info(f"Created user preferences file at {prefs_file}")
        print("Created user preferences file with default settings")
    else:
        logger.info(f"User preferences file already exists at {prefs_file}")
    
    # Create .env file for W&B if it doesn't exist
    env_file = project_dir / ".env"
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write("# Environment variables\n")
            f.write("# Set your W&B API key here for model training monitoring\n")
            f.write("WANDB_API_KEY=\n")
        
        logger.info(f"Created .env file at {env_file}")
        print("Created .env file for W&B configuration")
    else:
        logger.info(f".env file already exists at {env_file}")
    
    # Create finetune config if it doesn't exist
    finetune_config_file = project_dir / "finetune_config.json"
    if not finetune_config_file.exists():
        # Create default fine-tuning configuration
        finetune_config = {
            "base_model": "mistralai/Mistral-7B-Instruct-v0.2",
            "model_type": "causal_lm",
            "training": {
                "epochs": 3,
                "learning_rate": 3e-4,
                "batch_size": 8,
                "gradient_accumulation_steps": 2,
                "warmup_steps": 100,
                "weight_decay": 0.01,
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_epsilon": 1e-8,
                "max_grad_norm": 1.0
            },
            "lora": {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
            },
            "data": {
                "validation_split": 0.1,
                "max_length": 2048,
                "shuffle": True
            },
            "wandb": {
                "entity": "user",
                "project": "pete-connor-finetuning",
                "name": "pete-connor-model-run"
            }
        }
        
        # Write the config file
        with open(finetune_config_file, 'w') as f:
            json.dump(finetune_config, f, indent=2)
        
        logger.info(f"Created fine-tuning config file at {finetune_config_file}")
        print("Created fine-tuning configuration file")
    else:
        logger.info(f"Fine-tuning config file already exists at {finetune_config_file}")
    
    return True

def setup_test_data():
    """
    Create sample test data for demonstration purposes.
    """
    project_dir = Path(__file__).parent
    test_data_dir = project_dir / "data" / "test"
    test_data_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a sample test text file
    sample_text = """
    Multi-Format Content Generator: A Modern Approach to Content Creation
    
    Content creation for multiple platforms is a time-consuming process that requires adapting the same information to different formats, styles, and audience expectations. This project aims to simplify this process by providing a tool that can automatically generate platform-specific content from a single source.
    
    Key features include:
    - Template-based content generation
    - Sentiment analysis to adapt tone
    - Platform-specific formatting
    - Keyword extraction for relevant hashtags
    - Support for multiple input sources (text, documents, URLs)
    
    The applications of this tool extend to marketing teams, content creators, social media managers, and anyone who needs to publish consistent content across multiple platforms efficiently.
    """
    
    sample_file = test_data_dir / "sample_text.txt"
    with open(sample_file, 'w') as f:
        f.write(sample_text)
    
    logger.info(f"Created sample test file at {sample_file}")
    print(f"Created sample test data at {sample_file}")
    return True

def run_data_sync():
    """
    Run the data synchronization script if available.
    """
    sync_script = Path(__file__).parent / "sync_data.py"
    
    if sync_script.exists():
        try:
            logger.info("Running data synchronization...")
            
            # Method 1: Import and run
            spec = importlib.util.spec_from_file_location("sync_data", sync_script)
            sync_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sync_module)
            sync_module.sync_data_files()
            
            print("Data synchronization completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error running data synchronization: {str(e)}")
            print(f"Error running data synchronization: {str(e)}")
            return False
    else:
        logger.warning("Data synchronization script not found, skipping")
        return True

if __name__ == "__main__":
    print("Setting up data files...")
    data_success = setup_data_files()
    test_success = setup_test_data()
    sync_success = run_data_sync()
    
    if data_success and test_success and sync_success:
        print("\nSetup completed successfully!")
        print("You can now run the application with 'streamlit run src/app.py'")
    else:
        print("\nSetup encountered some errors.")
        print("Check the log for more information.")
