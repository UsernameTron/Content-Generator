"""
Setup script for Weights & Biases integration.
"""

import os
import sys
import logging
from dotenv import load_dotenv, set_key

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def setup_wandb():
    """
    Set up Weights & Biases integration by configuring the API key.
    """
    try:
        # Load existing .env file or create if it doesn't exist
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
        if not os.path.exists(env_path):
            with open(env_path, 'w') as f:
                f.write('# Environment variables for Multi-Platform Content Generator\n')
            logger.info(f"Created new .env file at {env_path}")
        
        # Load environment variables
        load_dotenv(env_path)
        
        # Check if WANDB_API_KEY is already set
        api_key = os.environ.get('WANDB_API_KEY')
        if api_key:
            print(f"Weights & Biases API key is already set.")
            update = input("Do you want to update it? (y/n): ").lower()
            if update != 'y':
                print("Setup complete. Using existing W&B API key.")
                return True
        
        # Prompt for API key
        api_key = input("Enter your Weights & Biases API key: ").strip()
        if not api_key:
            print("Error: API key cannot be empty.")
            return False
        
        # Set environment variable in .env file
        set_key(env_path, 'WANDB_API_KEY', api_key)
        
        # Set as environment variable for current session
        os.environ['WANDB_API_KEY'] = api_key
        
        print("\nW&B API key has been saved to .env file.")
        print("You can now use Weights & Biases for content generation monitoring.")
        
        # Setup project information
        project_name = input("Enter your W&B project name (default: multi-platform-content-generator): ").strip()
        if not project_name:
            project_name = "multi-platform-content-generator"
        
        set_key(env_path, 'WANDB_PROJECT', project_name)
        
        print(f"W&B project name set to: {project_name}")
        print("\nSetup complete!")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up W&B: {str(e)}")
        print(f"Error setting up W&B: {str(e)}")
        return False

def show_instructions():
    """
    Show instructions for using W&B with the application.
    """
    print("\n======================================================")
    print("  Weights & Biases Integration Instructions")
    print("======================================================")
    print("\n1. Create a W&B account at https://wandb.ai if you don't have one.")
    print("2. Get your API key from https://wandb.ai/settings")
    print("3. Run this setup script to save your API key")
    print("4. Use the application with W&B monitoring enabled")
    print("\nBenefits of W&B integration:")
    print("- Track content generation metrics")
    print("- Monitor sentiment analysis results")
    print("- Compare different writing styles and approaches")
    print("- View examples of generated content")
    print("======================================================\n")

if __name__ == "__main__":
    print("Setting up Weights & Biases integration...")
    show_instructions()
    success = setup_wandb()
    
    if success:
        print("\nYou can now run the application with W&B integration.")
        print("To view your results, go to: https://wandb.ai")
    else:
        print("\nW&B setup was not completed successfully.")
        print("You can still use the application, but W&B features will be disabled.")
        
    sys.exit(0 if success else 1)
